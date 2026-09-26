# Adapted from MetaClaw
"""
FastAPI proxy server for SkillClaw.

Intercepts LLM requests from Claw agents, injects skills into system
prompts, forwards to a real LLM API, and optionally collects PRM scores.
"""

from __future__ import annotations

import asyncio
import base64
import codecs
import copy
import hashlib
import json
import logging
import os
import random
import re
import struct
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .config import SkillClawConfig
from .prm_scorer import PRMScorer
from .protocols import anthropic_messages as anthropic_protocol
from .protocols import openai_responses as responses_protocol
from .skill_manager import SkillManager
from .utils import run_llm

logger = logging.getLogger(__name__)

_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_CYAN = "\033[36m"
_RESET = "\033[0m"

_NON_STANDARD_BODY_KEYS = {
    "session_id",
    "session_done",
    "turn_type",
    "_skillclaw_protocol",
    # Hermes may send this for reasoning-capable providers. SkillClaw exposes a
    # generic OpenAI-compatible proxy alias, and non-reasoning upstream chat
    # models such as gpt-4o reject this extension with HTTP 400. Strip it at
    # the proxy boundary instead of forwarding it blindly.
    "reasoning_effort",
}
_OPENAI_COMPAT_MAX_COMPLETION_TOKENS = 8192
# Floor for the prompt budget. Whatever completion size a client asks for,
# always keep at least this many estimated tokens of conversation so
# truncation stays active instead of being skipped by an oversized max_tokens.
_MIN_PROMPT_TOKENS = 1024
_PROTOCOL_ANTHROPIC_MESSAGES = "anthropic_messages"
_PROTOCOL_RESPONSES_COMPAT = "responses_compat"

# Bounds for the previous_response_id continuation store. It used to grow for
# the process lifetime, holding a full message list per turn, so a long-running
# proxy leaked memory in proportion to prompt size times turn count.
_RESPONSES_STORE_MAX_ENTRIES = 512
_RESPONSES_STORE_TTL_SECONDS = 3600.0
# Recently closed session ids, kept so a late request cannot resurrect a
# session's store entries. Bounded: ids are cheap and only needed to sweep.
_CLOSED_SESSION_MEMORY = 1024
# Bound on remembered pseudo-sessions for clients that send no session id.
_TUI_SESSION_MAX_ENTRIES = 512


# ------------------------------------------------------------------ #
# Helper utilities                                                     #
# ------------------------------------------------------------------ #


def _flatten_message_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # A part with a null or non-string `text` is legal input from a
        # misbehaving client; join() would raise on it, so filter to strings.
        parts = [
            text
            for item in content
            if isinstance(item, dict) and item.get("type") == "text" and isinstance((text := item.get("text")), str)
        ]
        return " ".join(parts) if parts else ""
    return str(content) if content is not None else ""


def _cap_completion_token_fields(body: dict[str, Any]) -> None:
    """Clamp oversized completion-token requests before forwarding upstream.

    Hermes can ask custom OpenAI-compatible routes for very large completions
    (for example 65536 tokens). Several upstream chat-completions models reject
    those requests immediately, and SkillClaw's retry loop makes that look like
    a hung smoke test. Keep the proxy request within a conservative
    OpenAI-compatible ceiling.
    """
    for key in ("max_tokens", "max_completion_tokens"):
        value = body.get(key)
        if value is None:
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > _OPENAI_COMPAT_MAX_COMPLETION_TOKENS:
            body[key] = _OPENAI_COMPAT_MAX_COMPLETION_TOKENS


def _openai_chat_requires_max_completion_tokens(model: str) -> bool:
    """Return True for OpenAI chat models that reject legacy max_tokens."""
    model = str(model or "").lower()
    return model.startswith(("gpt-5", "o1", "o3", "o4"))


def _normalize_openai_chat_token_fields(body: dict[str, Any]) -> None:
    """Translate legacy max_tokens for newer OpenAI chat-completions models."""
    if not _openai_chat_requires_max_completion_tokens(str(body.get("model") or "")):
        return
    if "max_tokens" in body and "max_completion_tokens" not in body:
        body["max_completion_tokens"] = body.pop("max_tokens")
    else:
        body.pop("max_tokens", None)


def _normalize_assistant_content_parts(content: list[dict]) -> tuple[str, list[dict]]:
    """Extract plain text and OpenAI-style tool_calls from assistant content parts."""
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    for i, item in enumerate(content):
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "text":
            text = item.get("text")
            if isinstance(text, str) and text:
                text_parts.append(text)
        elif item_type == "toolCall":
            name = item.get("name")
            args = item.get("arguments", {})
            if not isinstance(args, str):
                try:
                    args = json.dumps(args, ensure_ascii=False)
                except Exception:
                    args = "{}"
            tc_id = item.get("id") or f"call_{i}"
            tool_calls.append(
                {
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": name or "unknown_tool",
                        "arguments": args,
                    },
                }
            )
    return (" ".join(text_parts).strip(), tool_calls)


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_TOOL_HANDLE_RE = re.compile(r"^call_(?:kimi|xml)_\d+$")
_KIMI_TOOL_CALL_RE = re.compile(
    r"<\|tool_call_begin\|>\s*([a-zA-Z0-9_.-]+)(?::\d+)?\s*"
    r"<\|tool_call_argument_begin\|>\s*(\{.*?\})\s*"
    r"<\|tool_call_end\|>",
    re.DOTALL,
)
_QWEN_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_TOOL_ARGS_MAX_CHARS = 4_000
_TOOL_RESULT_CONTENT_MAX_CHARS = 4_000
_SESSION_IDLE_CLOSE_SECONDS = 180
_SESSION_SWEEP_INTERVAL_SECONDS = 15
_SHUTDOWN_DRAIN_TIMEOUT_SECONDS = 15
_VALID_TURN_TYPES = {"main", "side"}
_TRUE_STRINGS = {"1", "true", "yes", "on"}
_READ_TOOL_NAMES = {"read", "file_read", "read_file", "readfile"}
_HERMES_SKILL_READ_TOOL_NAMES = {"skill_view"}
_CLAUDE_CODE_SKILL_TOOL_NAMES = {"skill"}
_SKILL_WRITE_TOOL_NAMES = {
    "write",
    "file_write",
    "write_file",
    "writefile",
    "create_file",
    "edit",
    "edit_file",
    "replace",
    "replace_in_file",
    "append",
    "append_file",
    "patch",
    "apply_patch",
    "move",
    "rename",
    "mv",
}
_HERMES_SKILL_WRITE_TOOL_NAMES = {"skill_manage"}
_SHELL_TOOL_NAMES = {"shell", "exec", "bash", "terminal"}
_PATCH_PATH_RE = re.compile(r"^\*\*\* (?:Add|Update|Delete) File: (.+)$", re.MULTILINE)
_SHELL_SKILL_PATH_RE = re.compile(
    r"([~./A-Za-z0-9_\-][^\n\"'`]*?"
    r"(?:SKILL\.md|references/[^\s\"'`]+|scripts/[^\s\"'`]+|assets/[^\s\"'`]+|history/[^\s\"'`]+))"
)


def _extract_skill_names(items: list[Any] | None) -> set[str]:
    names: set[str] = set()
    for item in items or []:
        if isinstance(item, dict):
            raw = item.get("skill_name") or item.get("name") or item.get("skill")
        else:
            raw = item
        name = str(raw or "").strip()
        if name:
            names.add(name)
    return names


def _extract_modified_skill_names(turns: list[dict] | None) -> set[str]:
    names: set[str] = set()
    for turn in turns or []:
        if isinstance(turn, dict):
            names.update(_extract_skill_names(turn.get("modified_skills")))
    return names


def _is_unsupported_temperature_error(body_text: str) -> bool:
    """True when a 400 says the model or route rejects the ``temperature`` field.

    Providers word this several ways — ``"'temperature' is not supported with
    this model"``, ``"Unsupported value: 'temperature' does not support 0"``,
    ``"Unsupported parameter: 'temperature'"`` — so match on the intent rather
    than one exact sentence.
    """
    if not body_text:
        return False
    lowered = body_text.lower()
    if "temperature" not in lowered:
        return False
    return any(
        marker in lowered
        for marker in (
            "not supported",
            "unsupported",
            "does not support",
            "unrecognized request argument",
            "unknown parameter",
            "unexpected keyword",
        )
    )


def _coerce_int(value: Any, default: int) -> int:
    """int() with a fallback, so a client-supplied "auto" cannot 500 the proxy."""
    if value is None or isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _is_retryable_status(status_code: int) -> bool:
    """Whether a failed upstream call is worth sending again.

    Only transient conditions qualify: a retryable status, or a transport-level
    failure. A 400/401/403/404/409/422 is deterministic — the same body will be
    rejected again — so retrying it just multiplies latency and rejected
    requests. 408 and 429 are the client-error statuses that do clear on their
    own, and 5xx is the usual transient server class.
    """
    return status_code in {408, 429} or 500 <= status_code < 600


def _retry_after_seconds(response: Any, fallback: float) -> float:
    """Honour a ``Retry-After`` header, falling back to exponential backoff."""
    raw = None
    try:
        raw = response.headers.get("Retry-After")
    except Exception:
        raw = None
    if raw is not None:
        try:
            return max(0.0, min(float(str(raw).strip()), 60.0))
        except (TypeError, ValueError):
            pass
    return fallback


def _llm_request_timeout_seconds() -> float:
    raw = str(os.environ.get("SKILLCLAW_LLM_REQUEST_TIMEOUT_S", "120")).strip()
    try:
        timeout = float(raw)
    except ValueError:
        return 120.0
    return timeout if timeout > 0 else 120.0


def _resolve_turn_type(
    header_turn_type: Optional[str],
    body_turn_type: Any,
    *,
    default: str = "main",
) -> str:
    """Resolve request turn_type safely.

    Defaults to ``main`` to avoid silently dropping record/PRM paths when
    clients include a session id but forget to provide turn_type.
    """
    if default not in _VALID_TURN_TYPES:
        default = "main"
    candidate = header_turn_type if header_turn_type is not None else body_turn_type
    raw = str(candidate or "").strip().lower()
    if not raw:
        return default
    if raw in _VALID_TURN_TYPES:
        return raw
    logger.warning("[SessionDetect] invalid turn_type=%r; fallback=%s", raw, default)
    return default


def _resolve_session_done(
    header_session_done: Optional[str],
    body_session_done: Any,
) -> bool:
    """Resolve session_done from header or body."""
    candidate = header_session_done if header_session_done is not None else body_session_done
    if isinstance(candidate, bool):
        return candidate
    if candidate is None:
        return False
    return str(candidate).strip().lower() in _TRUE_STRINGS


def _looks_like_session_title_response(content: str) -> bool:
    """Return True for Claude Code's internal generate_session_title response."""
    text = str(content or "").strip()
    if not text or len(text) > 500:
        return False
    try:
        parsed = json.loads(text)
    except Exception:
        return False
    if not isinstance(parsed, dict) or set(parsed.keys()) - {"title"}:
        return False
    title = parsed.get("title")
    return isinstance(title, str) and bool(title.strip())


def _classify_raw_turn_kind(protocol: str, content: str, tool_calls: list[dict]) -> str:
    """Classify recorded raw/main turns for user-turn cadence decisions."""
    if tool_calls:
        return "tool_use"
    if protocol == _PROTOCOL_ANTHROPIC_MESSAGES and _looks_like_session_title_response(content):
        return "session_title"
    return "final"


def _is_user_turn_boundary(raw_turn_kind: str) -> bool:
    """Only final assistant responses advance the user-visible turn counter."""
    return raw_turn_kind == "final"


def _normalize_tool_name(raw_name: str, args_raw: str) -> str:
    """
    Normalize tool names from model output.
    Fixes common drift where a call handle (e.g. call_kimi_0) is emitted as
    function name instead of the actual tool name.
    """
    name = (raw_name or "").strip()
    if name.startswith("functions."):
        name = name.split(".", 1)[1]
    if not _TOOL_HANDLE_RE.fullmatch(name):
        return name or "unknown_tool"

    try:
        args_obj = json.loads(args_raw or "{}")
    except Exception:
        args_obj = {}
    if isinstance(args_obj, dict):
        if isinstance(args_obj.get("command"), str) and args_obj.get("command"):
            return "exec"
        if isinstance(args_obj.get("sessionId"), str) and args_obj.get("sessionId"):
            return "process"
    return "unknown_tool"


def _normalize_tool_call_name(raw_name: str) -> str:
    """Strip transport-specific prefixes from a tool name."""
    name = str(raw_name or "").strip()
    if name.startswith("functions."):
        return name.split(".", 1)[1]
    return name


def _deduplicate_paths(paths: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for path in paths:
        clean = str(path or "").strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        out.append(clean)
    return out


def _looks_like_path(value: str) -> bool:
    text = str(value or "").strip()
    if not text or text in {".", ".."}:
        return False
    return "/" in text or "\\" in text or text.startswith("~") or text.endswith("SKILL.md")


def _extract_skill_paths_from_patch(raw_text: str) -> list[str]:
    return _deduplicate_paths(
        [match.group(1).strip() for match in _PATCH_PATH_RE.finditer(str(raw_text or "")) if match.group(1).strip()]
    )


def _extract_skill_paths_from_shell(command: str) -> list[str]:
    return _deduplicate_paths(
        [
            match.group(1).strip()
            for match in _SHELL_SKILL_PATH_RE.finditer(str(command or ""))
            if match.group(1).strip()
        ]
    )


def _extract_skill_paths_from_args_dict(args: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for key in (
        "path",
        "file",
        "file_path",
        "target",
        "destination",
        "dest",
        "to",
        "source",
        "src",
        "old_path",
        "new_path",
    ):
        value = args.get(key)
        if isinstance(value, str) and _looks_like_path(value):
            paths.append(value.strip())

    raw_paths = args.get("paths")
    if isinstance(raw_paths, list):
        for item in raw_paths:
            if isinstance(item, str) and _looks_like_path(item):
                paths.append(item.strip())
    return _deduplicate_paths(paths)


def _extract_skill_paths_from_tool_call(tool_call: dict) -> tuple[str, list[str]]:
    func = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
    tool_name = _normalize_tool_call_name(func.get("name") or "")
    args_raw = func.get("arguments", "{}")
    if not isinstance(args_raw, str):
        try:
            args_raw = json.dumps(args_raw, ensure_ascii=False)
        except Exception:
            args_raw = "{}"

    paths: list[str] = []
    args_obj: Any = None
    try:
        args_obj = json.loads(args_raw)
    except Exception:
        args_obj = None

    if isinstance(args_obj, dict):
        paths.extend(_extract_skill_paths_from_args_dict(args_obj))
        if tool_name.lower() in _SHELL_TOOL_NAMES:
            command = str(args_obj.get("command") or args_obj.get("cmd") or "")
            paths.extend(_extract_skill_paths_from_shell(command))

    if tool_name.lower() in {"apply_patch", "patch"}:
        paths.extend(_extract_skill_paths_from_patch(args_raw))
    elif tool_name.lower() in _SHELL_TOOL_NAMES:
        paths.extend(_extract_skill_paths_from_shell(args_raw))

    return tool_name, _deduplicate_paths(paths)


def _extract_hermes_skill_name_from_tool_call(tool_call: dict) -> tuple[str, str, str]:
    """Extract Hermes-native skill name + relative file path from skill calls."""
    func = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
    tool_name = _normalize_tool_call_name(func.get("name") or "")
    args_raw = func.get("arguments", "{}")
    if not isinstance(args_raw, str):
        try:
            args_raw = json.dumps(args_raw, ensure_ascii=False)
        except Exception:
            args_raw = "{}"

    try:
        args_obj = json.loads(args_raw)
    except Exception:
        args_obj = {}

    if not isinstance(args_obj, dict):
        return tool_name, "", ""

    rel_path = ""
    for key in ("file_path", "path"):
        value = args_obj.get(key)
        if isinstance(value, str) and value.strip():
            rel_path = value.strip()
            break
    for key in ("skill_name", "name", "skill"):
        value = args_obj.get(key)
        if isinstance(value, str) and value.strip():
            return tool_name, value.strip(), rel_path
    return tool_name, "", rel_path


def _resolve_skill_reference(
    path: str,
    skill_path_map: dict[str, dict[str, str]],
) -> dict[str, str]:
    expanded = os.path.expanduser(str(path or "").strip())
    real_path = os.path.realpath(expanded) if expanded else ""
    skill_info = (
        skill_path_map.get(real_path) or skill_path_map.get(expanded) or skill_path_map.get(str(path or "").strip())
    )
    if skill_info:
        return {
            "skill_id": str(skill_info.get("skill_id", "") or ""),
            "skill_name": str(skill_info.get("skill_name", "") or ""),
            "path": str(path or "").strip(),
        }
    return {
        "skill_id": "",
        "skill_name": "",
        "path": str(path or "").strip(),
    }


def _resolve_skill_reference_by_name(
    skill_name: str,
    skill_path_map: dict[str, dict[str, str]],
    rel_path: str = "",
) -> dict[str, str]:
    clean_name = str(skill_name or "").strip()
    if not clean_name:
        return {"skill_id": "", "skill_name": "", "path": ""}
    normalized_rel = str(rel_path or "").strip().replace("\\", "/").lstrip("./")
    if normalized_rel:
        suffix = f"/{normalized_rel}"
        for path, skill_info in skill_path_map.items():
            if str(skill_info.get("skill_name", "") or "").strip() != clean_name:
                continue
            candidate = str(path or "").replace("\\", "/")
            if candidate.endswith(suffix) or candidate == normalized_rel:
                return {
                    "skill_id": str(skill_info.get("skill_id", "") or ""),
                    "skill_name": clean_name,
                    "path": str(path or ""),
                }
    for path, skill_info in skill_path_map.items():
        if str(skill_info.get("skill_name", "") or "").strip() == clean_name:
            return {
                "skill_id": str(skill_info.get("skill_id", "") or ""),
                "skill_name": clean_name,
                "path": str(path or ""),
            }
    return {"skill_id": "", "skill_name": clean_name, "path": ""}


def _extract_tool_calls_from_text(text: str) -> tuple[str, list[dict]]:
    """
    Parse tool-call tags embedded in assistant text into OpenAI-style tool_calls.
    Supports Kimi markers and Qwen <tool_call> wrappers.
    """
    if not text:
        return "", []

    tool_calls: list[dict] = []

    for i, m in enumerate(_KIMI_TOOL_CALL_RE.finditer(text)):
        raw_name = (m.group(1) or "").strip()
        args_raw = (m.group(2) or "{}").strip()
        tool_name = _normalize_tool_name(raw_name, args_raw)
        try:
            args_obj = json.loads(args_raw)
            args_str = json.dumps(args_obj, ensure_ascii=False)
        except Exception:
            args_str = args_raw if args_raw else "{}"
        tool_calls.append(
            {
                "id": f"call_kimi_{i}",
                "type": "function",
                "function": {"name": tool_name or "unknown_tool", "arguments": args_str},
            }
        )

    for i, m in enumerate(_QWEN_TOOL_CALL_RE.finditer(text), start=len(tool_calls)):
        payload_raw = (m.group(1) or "").strip()
        try:
            payload = json.loads(payload_raw)
        except Exception:
            continue
        name = (
            payload.get("name") or payload.get("tool_name") or payload.get("function", {}).get("name") or "unknown_tool"
        )
        args = payload.get("arguments") or payload.get("function", {}).get("arguments") or {}
        if not isinstance(args, str):
            try:
                args = json.dumps(args, ensure_ascii=False)
            except Exception:
                args = "{}"
        name = _normalize_tool_name(str(name), args)
        tool_calls.append(
            {
                "id": f"call_xml_{i}",
                "type": "function",
                "function": {"name": name, "arguments": args},
            }
        )

    clean = text
    clean = _THINK_RE.sub("", clean)
    clean = clean.replace("</think>", "")
    # Keep tool call data only in structured field; strip markup from plain text.
    clean = re.sub(r"<\|tool_call_begin\|>.*?<\|tool_call_end\|>", "", clean, flags=re.DOTALL)
    clean = re.sub(r"<\|tool_calls_section_begin\|>.*?<\|tool_calls_section_end\|>", "", clean, flags=re.DOTALL)
    clean = _QWEN_TOOL_CALL_RE.sub("", clean)
    clean = clean.strip()
    return clean, tool_calls


def _assistant_message_has_tool_calls(message: dict[str, Any]) -> bool:
    raw_tool_calls = message.get("tool_calls")
    if isinstance(raw_tool_calls, list) and raw_tool_calls:
        return True

    raw_content = message.get("content")
    if isinstance(raw_content, list):
        _, part_tool_calls = _normalize_assistant_content_parts(raw_content)
        return bool(part_tool_calls)
    if isinstance(raw_content, str) and raw_content:
        _, text_tool_calls = _extract_tool_calls_from_text(raw_content)
        return bool(text_tool_calls)
    return False


def _restore_missing_reasoning_content(
    messages: list[dict[str, Any]],
    prior_turns: list[dict[str, Any]],
) -> int:
    """Backfill reasoning_content for prior assistant tool-call messages."""
    assistant_tool_indices = [
        idx
        for idx, msg in enumerate(messages)
        if isinstance(msg, dict) and msg.get("role") == "assistant" and _assistant_message_has_tool_calls(msg)
    ]
    prior_tool_turns = [turn for turn in prior_turns if isinstance(turn, dict) and turn.get("tool_calls")]
    if not assistant_tool_indices or not prior_tool_turns:
        return 0

    # Pair by tool_call id, not by position: zipping two reversed lists
    # mis-attributed every reasoning block when the request carried more
    # assistant tool-call messages than the proxy had recorded, so one call's
    # chain-of-thought was sent with a different call and the real one dropped.
    turns_by_id: dict[str, list[dict[str, Any]]] = {}
    for turn in prior_tool_turns:
        reasoning = str(turn.get("reasoning_content") or "").strip()
        if not reasoning:
            continue
        for call_id in _turn_tool_call_ids(turn):
            turns_by_id.setdefault(call_id, []).append(turn)

    restored = 0
    used: set[int] = set()
    for msg_idx in reversed(assistant_tool_indices):
        msg = messages[msg_idx]
        if msg.get("reasoning_content"):
            continue
        for call_id in _message_tool_call_ids(msg):
            candidates = turns_by_id.get(call_id) or []
            turn = next((t for t in candidates if id(t) not in used), None)
            if turn is None:
                continue
            used.add(id(turn))
            messages[msg_idx] = {**msg, "reasoning_content": str(turn["reasoning_content"]).strip()}
            restored += 1
            break
    return restored


def _turn_tool_call_ids(turn: dict[str, Any]) -> list[str]:
    """Tool-call ids recorded on a session turn."""
    ids: list[str] = []
    raw = turn.get("tool_calls")
    if isinstance(raw, list):
        for call in raw:
            if isinstance(call, dict):
                call_id = str(call.get("id") or call.get("tool_call_id") or "").strip()
                if call_id:
                    ids.append(call_id)
    single = str(turn.get("tool_call_id") or "").strip()
    if single:
        ids.append(single)
    return ids


def _deduplicate_tool_calls(tool_calls: list[dict]) -> list[dict]:
    """Deduplicate tool calls while preserving order.

    Only an explicit id identifies a duplicate. Falling back to
    (name, arguments) merged two genuinely distinct parallel calls that a
    provider emitted without ids — Hermes' own text-parsed tool calls
    (``call_kimi_N``/``call_xml_N``) and several upstreams do omit them.
    """
    deduped: list[dict] = []
    seen_ids: set[str] = set()
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        tc_id = str(tc.get("id") or "").strip()
        if tc_id:
            if tc_id in seen_ids:
                continue
            seen_ids.add(tc_id)
        deduped.append(tc)
    return deduped


def _normalize_messages_for_template(messages: list[dict]) -> list[dict]:
    """Normalize OpenClaw-style messages into chat-template-compatible format."""
    out = []
    for msg in messages:
        m = dict(msg)
        role = m.get("role")

        if role == "developer":
            m["role"] = "system"
            role = "system"

        # OpenClaw tool result message → OpenAI tool message
        if role == "toolResult":
            tool_msg: dict[str, Any] = {
                "role": "tool",
                "content": _flatten_message_content(m.get("content")),
            }
            tc_id = m.get("toolCallId") or m.get("tool_call_id")
            if tc_id:
                tool_msg["tool_call_id"] = tc_id
            tool_name = m.get("toolName") or m.get("name")
            if tool_name:
                tool_msg["name"] = tool_name
            out.append(tool_msg)
            continue

        # assistant content parts may contain text + toolCall blocks
        raw = m.get("content")
        if role == "assistant" and isinstance(raw, list):
            text, tool_calls = _normalize_assistant_content_parts(raw)
            m["content"] = text
            if tool_calls:
                m["tool_calls"] = tool_calls
        elif not isinstance(raw, str) and raw is not None:
            m["content"] = _flatten_message_content(raw)

        out.append(m)
    return out


def _extract_last_user_instruction(messages: list[dict]) -> str:
    """Return the most recent user message text from the current turn context."""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            text = _flatten_message_content(msg.get("content"))
            if text:
                return text
    return ""


_ERROR_PATTERNS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(
            r"exited with code (?!0\b)\d+|exit code (?!0\b)\d+|exit status (?!0\b)\d+",
            re.IGNORECASE,
        ),
        "exit_code",
    ),
    (re.compile(r"Traceback \(most recent call last\)|\.py\", line \d+", re.IGNORECASE), "traceback"),
    (re.compile(r"Permission denied|EACCES|PermissionError", re.IGNORECASE), "permission"),
    (re.compile(r"No such file|FileNotFoundError|ENOENT|not found", re.IGNORECASE), "not_found"),
    (re.compile(r"command not found|not recognized as|is not recognized", re.IGNORECASE), "command_not_found"),
    (re.compile(r"timed?\s*out|TimeoutError|ETIMEDOUT", re.IGNORECASE), "timeout"),
    (re.compile(r"(?:^|\W)(?:Error|Exception):\s", re.MULTILINE), "generic_error"),
]


def _classify_tool_error(content: str) -> tuple[bool, str | None]:
    """Return (has_error, error_type) by matching content against known patterns."""
    for pattern, error_type in _ERROR_PATTERNS:
        if pattern.search(content):
            return True, error_type
    return False, None


def _extract_recent_tool_results(messages: list[dict]) -> list[dict]:
    """Extract tool results from the most recent tool-call round in messages.

    Scans backwards from the end of *messages*, collecting all consecutive
    tool / toolResult messages that appear after the last assistant message.
    Returns a list of summary dicts suitable for skill feedback tracking.
    """
    results: list[dict] = []
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        if role in ("toolResult", "tool"):
            content = _flatten_message_content(msg.get("content"))
            tool_name = msg.get("toolName") or msg.get("name") or msg.get("tool_name") or "unknown"
            has_error, error_type = _classify_tool_error(content)
            results.append(
                {
                    "tool_name": tool_name,
                    "tool_call_id": (msg.get("toolCallId") or msg.get("tool_call_id") or ""),
                    "content": content[:_TOOL_RESULT_CONTENT_MAX_CHARS],
                    "has_error": has_error,
                    "error_type": error_type,
                }
            )
        elif role == "user":
            continue
        else:
            break
    results.reverse()
    return results


def _extract_recent_tool_result_messages(messages: list[dict]) -> list[dict]:
    """Extract raw tool result messages from the most recent tool round.

    This preserves the original payload shape so cloud sessions can retain a
    complete tool-execution snapshot for future analysis. No truncation or
    error classification is applied here.
    """
    results: list[dict] = []
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        if role in ("toolResult", "tool"):
            try:
                results.append(json.loads(json.dumps(msg, ensure_ascii=False)))
            except Exception:
                results.append(dict(msg))
        elif role == "user":
            continue
        else:
            break
    results.reverse()
    return results


def _assemble_streaming_chat_completion(
    events: list[dict[str, Any]],
    *,
    fallback_model: str,
) -> dict[str, Any]:
    """Collapse OpenAI-style SSE chat chunks into a single response dict."""
    import time

    builders: dict[int, dict[str, Any]] = {}
    response_id = ""
    response_model = fallback_model
    response_created = int(time.time())
    usage: dict[str, Any] = {}

    for event in events:
        if not isinstance(event, dict):
            continue
        response_id = str(event.get("id") or response_id)
        response_model = str(event.get("model") or response_model)
        created = event.get("created")
        if isinstance(created, int):
            response_created = created
        if isinstance(event.get("usage"), dict):
            usage = dict(event["usage"])

        for choice in event.get("choices", []) or []:
            if not isinstance(choice, dict):
                continue
            index = _coerce_int(choice.get("index"), 0)
            entry = builders.setdefault(
                index,
                {
                    "role": "assistant",
                    "content_parts": [],
                    "tool_calls": {},
                    "finish_reason": None,
                },
            )
            delta = choice.get("delta") or {}
            if isinstance(delta.get("role"), str):
                entry["role"] = delta["role"]

            content = delta.get("content")
            if isinstance(content, str):
                entry["content_parts"].append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and isinstance(item.get("text"), str):
                        entry["content_parts"].append(item["text"])

            for tc in delta.get("tool_calls", []) or []:
                if not isinstance(tc, dict):
                    continue
                # A provider may serialise an explicit `index: null`; int(None)
                # used to raise and turn the whole request into a 502.
                tc_index = _coerce_int(tc.get("index"), 0)
                tool_entry = entry["tool_calls"].setdefault(
                    tc_index,
                    {
                        "id": tc.get("id") or f"call_{tc_index}",
                        "type": tc.get("type", "function"),
                        "function": {"name": "", "arguments": ""},
                    },
                )
                if tc.get("id"):
                    tool_entry["id"] = tc["id"]
                if tc.get("type"):
                    tool_entry["type"] = tc["type"]
                fn = tc.get("function") or {}
                if isinstance(fn.get("name"), str):
                    tool_entry["function"]["name"] += fn["name"]
                if isinstance(fn.get("arguments"), str):
                    tool_entry["function"]["arguments"] += fn["arguments"]

            finish_reason = choice.get("finish_reason")
            if finish_reason is not None:
                entry["finish_reason"] = finish_reason

    choices: list[dict[str, Any]] = []
    for index in sorted(builders):
        entry = builders[index]
        message: dict[str, Any] = {
            "role": entry["role"],
            "content": "".join(entry["content_parts"]),
        }
        if entry["tool_calls"]:
            message["tool_calls"] = [entry["tool_calls"][i] for i in sorted(entry["tool_calls"])]
        choices.append(
            {
                "index": index,
                "message": message,
                "finish_reason": entry["finish_reason"] or "stop",
            }
        )

    return {
        "id": response_id or f"chatcmpl-stream-{response_created}",
        "object": "chat.completion",
        "created": response_created,
        "model": response_model,
        "choices": choices
        or [
            {
                "index": 0,
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
    }


class _SseStreamError(Exception):
    """An error frame arrived inside an otherwise-healthy SSE stream.

    OpenAI-compatible streams can report a mid-generation failure as a
    ``data: {"error": {...}}`` frame and then close. Treating the partial text
    as a normal completion reports a truncated answer as a success, so this is
    raised instead of letting the assembler finish silently.
    """

    def __init__(self, message: str, code: str = "upstream_error"):
        super().__init__(message)
        self.message = message
        self.code = code


def _stream_error_event(error: Exception, response_id: str = "", model: str = "") -> dict[str, Any]:
    """Build a client-facing error frame for an already-started SSE stream."""
    message = str(error) or error.__class__.__name__
    code = getattr(error, "code", "upstream_error")
    if getattr(error, "status_code", None) is not None:
        detail: dict[str, Any] = f"Upstream LLM error: {error}"
    else:
        detail = {"message": message, "type": "upstream_error", "code": code}
    return {
        "error": {
            "message": message,
            "type": "upstream_error",
            "code": code,
            "param": None,
            "detail": detail,
        },
        "id": response_id,
        "object": "chat.completion.chunk",
        "model": model,
    }


def _sse_chunk(payload: dict[str, Any]) -> str:
    """Serialize one SSE data frame."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _collect_sse_chat_events(response) -> list[dict[str, Any]]:
    """Read SSE `data:` lines from a streaming chat completion response.

    Raises :class:`_SseStreamError` when a frame carries an ``error`` object
    so the caller can fail the request instead of returning a partial answer
    as a normal completion.
    """
    events: list[dict[str, Any]] = []
    async for line in response.aiter_lines():
        if not line:
            continue
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if isinstance(event.get("error"), (dict, str)):
            error = event["error"]
            if isinstance(error, dict):
                message = str(error.get("message") or json.dumps(error, ensure_ascii=False)[:200])
                code = str(error.get("code") or error.get("type") or "upstream_error")
            else:
                message, code = str(error), "upstream_error"
            raise _SseStreamError(message, code)
        events.append(event)
    return events


def _build_tool_summaries(tool_calls: list[dict]) -> list[dict]:
    """Build tool summary dicts from the model's tool_calls.

    Extracts the tool name and key arguments (``command`` for shell-like
    tools, ``path`` for file-based tools) into a compact format for the
    turn record.  ``has_error`` defaults to ``False`` and is merged later
    when actual tool results arrive.
    """
    summaries: list[dict] = []
    for tc in tool_calls:
        func = tc.get("function", {})
        name = _normalize_tool_call_name(func.get("name", "unknown"))
        args_raw = func.get("arguments", "{}")
        if not isinstance(args_raw, str):
            try:
                args_raw = json.dumps(args_raw, ensure_ascii=False)
            except Exception:
                args_raw = "{}"
        try:
            args = json.loads(args_raw)
        except Exception:
            args = {}
        _, skill_paths = _extract_skill_paths_from_tool_call(tc)

        summary: dict[str, Any] = {
            "tool_name": name,
            "tool_call_id": str(tc.get("id") or ""),
            "arguments": args_raw[:_TOOL_ARGS_MAX_CHARS],
            "has_error": False,
        }

        if name.lower() in _SHELL_TOOL_NAMES:
            cmd = str(args.get("command") or args.get("cmd") or "")
            if cmd:
                summary["command"] = cmd[:_TOOL_ARGS_MAX_CHARS]

        path = str(args.get("path") or args.get("file") or args.get("file_path") or "")
        if path:
            summary["path"] = path
        elif skill_paths:
            summary["path"] = skill_paths[0]

        summaries.append(summary)
    return summaries


def _extract_read_skills_from_tool_calls(
    tool_calls: list[dict],
    skill_path_map: dict[str, dict[str, str]],
) -> list[dict]:
    """Identify which skill bundle files were read from the model's tool_calls.

    Returns a list of ``{"skill_id": ..., "skill_name": ...}`` dicts for
    each ``read`` tool call whose ``path`` argument points inside a skill.
    """
    read_skills: list[dict] = []
    seen_ids: set[str] = set()
    for tc in tool_calls:
        tool_name, skill_paths = _extract_skill_paths_from_tool_call(tc)
        normalized = tool_name.lower()
        if normalized in _HERMES_SKILL_READ_TOOL_NAMES:
            _, skill_name, rel_path = _extract_hermes_skill_name_from_tool_call(tc)
            skill_ref = _resolve_skill_reference_by_name(skill_name, skill_path_map, rel_path)
            dedupe_key = skill_ref.get("skill_id") or skill_ref.get("skill_name")
            if dedupe_key and dedupe_key not in seen_ids:
                read_skills.append(skill_ref)
                seen_ids.add(dedupe_key)
            continue
        if normalized in _CLAUDE_CODE_SKILL_TOOL_NAMES:
            _, skill_name, rel_path = _extract_hermes_skill_name_from_tool_call(tc)
            skill_ref = _resolve_skill_reference_by_name(skill_name, skill_path_map, rel_path)
            dedupe_key = skill_ref.get("skill_id") or skill_ref.get("skill_name")
            if dedupe_key and dedupe_key not in seen_ids:
                read_skills.append(skill_ref)
                seen_ids.add(dedupe_key)
            continue
        if normalized not in _READ_TOOL_NAMES:
            continue
        for path in skill_paths:
            skill_ref = _resolve_skill_reference(path, skill_path_map)
            if not skill_ref.get("skill_id") and not skill_ref.get("skill_name"):
                continue
            dedupe_key = skill_ref.get("skill_id") or skill_ref.get("path") or skill_ref.get("skill_name")
            if not dedupe_key or dedupe_key in seen_ids:
                continue
            read_skills.append(skill_ref)
            seen_ids.add(dedupe_key)

    return read_skills


def _extract_modified_skills_from_tool_calls(
    tool_calls: list[dict],
    skill_path_map: dict[str, dict[str, str]],
) -> list[dict]:
    """Identify skill bundle files the model attempted to write or update."""
    modified_skills: list[dict] = []
    seen_ids: set[str] = set()
    for tc in tool_calls:
        tool_name, skill_paths = _extract_skill_paths_from_tool_call(tc)
        normalized = tool_name.lower()
        if normalized in _READ_TOOL_NAMES:
            continue
        if normalized in _HERMES_SKILL_WRITE_TOOL_NAMES:
            _, skill_name, rel_path = _extract_hermes_skill_name_from_tool_call(tc)
            skill_ref = _resolve_skill_reference_by_name(skill_name, skill_path_map, rel_path)
            dedupe_key = skill_ref.get("skill_id") or skill_ref.get("skill_name")
            if dedupe_key and dedupe_key not in seen_ids:
                modified_skills.append({**skill_ref, "action": normalized})
                seen_ids.add(dedupe_key)
            continue
        if normalized not in _SKILL_WRITE_TOOL_NAMES and normalized not in _SHELL_TOOL_NAMES:
            continue
        for path in skill_paths:
            skill_ref = _resolve_skill_reference(path, skill_path_map)
            if not skill_ref.get("skill_id") and not skill_ref.get("skill_name"):
                continue
            dedupe_key = skill_ref.get("skill_id") or skill_ref.get("path") or skill_ref.get("skill_name")
            if not dedupe_key or dedupe_key in seen_ids:
                continue
            modified_skills.append(
                {
                    **skill_ref,
                    "action": "shell" if normalized in _SHELL_TOOL_NAMES else normalized,
                }
            )
            seen_ids.add(dedupe_key)
    return modified_skills


def _merge_tool_error_info(
    turn_record: dict,
    tool_results: list[dict],
    raw_tool_results: list[dict] | None = None,
) -> None:
    """Merge error information from tool results into the turn record.

    Matches tool results to the ``tool_results`` summaries built from tool
    calls (by position).  Updates ``has_error``, ``error_type``, and
    ``content`` on matching entries, then rebuilds ``tool_errors``.
    ``raw_tool_results`` preserves the original tool payloads for cloud upload.
    """
    summaries = turn_record.get("tool_results", [])
    observations: list[dict] = []

    if raw_tool_results is not None:
        raw_snapshot: list[dict] = []
        for item in raw_tool_results:
            if not isinstance(item, dict):
                continue
            try:
                raw_snapshot.append(json.loads(json.dumps(item, ensure_ascii=False)))
            except Exception:
                raw_snapshot.append(dict(item))
        turn_record["tool_results_raw"] = raw_snapshot
    else:
        turn_record.setdefault("tool_results_raw", [])

    for i, result in enumerate(tool_results):
        obs: dict[str, Any] = {
            "tool_name": result.get("tool_name", "unknown"),
            "tool_call_id": result.get("tool_call_id", ""),
            "has_error": bool(result.get("has_error", False)),
        }
        if result.get("error_type"):
            obs["error_type"] = result["error_type"]
        content = result.get("content", "")
        if content:
            obs["content"] = str(content)[:_TOOL_RESULT_CONTENT_MAX_CHARS]
        observations.append(obs)

        if i < len(summaries):
            summaries[i]["has_error"] = bool(result.get("has_error", False))
            summaries[i]["tool_name"] = result.get("tool_name", summaries[i].get("tool_name", "unknown"))
            if result.get("tool_call_id"):
                summaries[i]["tool_call_id"] = result["tool_call_id"]
            if result.get("error_type"):
                summaries[i]["error_type"] = result["error_type"]
            else:
                summaries[i].pop("error_type", None)
            content = result.get("content", "")
            if content:
                summaries[i]["content"] = str(content)[:_TOOL_RESULT_CONTENT_MAX_CHARS]
            else:
                summaries[i].pop("content", None)
        else:
            entry: dict[str, Any] = {
                "tool_name": result.get("tool_name", "unknown"),
                "tool_call_id": result.get("tool_call_id", ""),
                "has_error": bool(result.get("has_error", False)),
            }
            if result.get("error_type"):
                entry["error_type"] = result["error_type"]
            content = result.get("content", "")
            if content:
                entry["content"] = str(content)[:_TOOL_RESULT_CONTENT_MAX_CHARS]
            summaries.append(entry)

    turn_record["tool_observations"] = observations
    turn_record["tool_errors"] = [
        {
            "tool_name": s.get("tool_name", "unknown"),
            **({"tool_call_id": s["tool_call_id"]} if s.get("tool_call_id") else {}),
            **({"error_type": s["error_type"]} if s.get("error_type") else {}),
            **({"content": s["content"]} if s.get("content") else {}),
        }
        for s in summaries
        if s.get("has_error")
    ]


def _rewrite_new_session_bootstrap_prompt(messages: list[dict]) -> tuple[list[dict], int]:
    """Rewrite OpenClaw /new bootstrap user prompt to a safer variant.

    Some upstream providers over-trigger policy filters on the stock bootstrap
    text ("A new session was started via /new or /reset ..."). This keeps
    behavior while avoiding brittle phrasing.
    """
    rewritten = 0
    out: list[dict] = []
    for msg in messages:
        if not isinstance(msg, dict):
            out.append(msg)
            continue
        if msg.get("role") != "user":
            out.append(msg)
            continue
        text = _flatten_message_content(msg.get("content"))
        lowered = text.lower()
        if "a new session was started via /new or /reset" in lowered:
            out.append(
                {
                    **msg,
                    "content": (
                        "A new chat session just started. "
                        "Greet the user briefly in 1-3 sentences and ask what they want to do."
                    ),
                }
            )
            rewritten += 1
            continue
        out.append(msg)
    return out, rewritten


# ------------------------------------------------------------------ #
# Protocol compatibility wrappers                                      #
# ------------------------------------------------------------------ #


def _anthropic_to_openai_body(body: dict[str, Any]) -> dict[str, Any]:
    return anthropic_protocol.to_openai_body(body)


def _anthropic_request_tool_names(body: dict[str, Any]) -> set[str]:
    tool_names: set[str] = set()
    tools = body.get("tools")
    if not isinstance(tools, list):
        return tool_names
    for item in tools:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if name:
            tool_names.add(name)
    return tool_names


_IMAGE_TOKEN_ESTIMATE = 1600


def _data_url_bytes(url: str) -> bytes | None:
    if not url.startswith("data:") or "," not in url:
        return None
    header, data = url.split(",", 1)
    if ";base64" not in header:
        return None
    try:
        return base64.b64decode(data, validate=False)
    except Exception:
        return None


def _image_dimensions_from_bytes(data: bytes) -> tuple[int, int] | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        width, height = struct.unpack(">II", data[16:24])
        return (width, height) if width > 0 and height > 0 else None
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        if len(data) >= 10:
            width, height = struct.unpack("<HH", data[6:10])
            return (width, height) if width > 0 and height > 0 else None
        return None
    if data.startswith(b"RIFF") and len(data) >= 30 and data[8:12] == b"WEBP":
        if data[12:16] == b"VP8X":
            width = int.from_bytes(data[24:27], "little") + 1
            height = int.from_bytes(data[27:30], "little") + 1
            return (width, height) if width > 0 and height > 0 else None
        if data[12:16] == b"VP8 " and len(data) >= 30:
            width = struct.unpack("<H", data[26:28])[0] & 0x3FFF
            height = struct.unpack("<H", data[28:30])[0] & 0x3FFF
            return (width, height) if width > 0 and height > 0 else None
    if data.startswith(b"\xff\xd8"):
        index = 2
        while index + 9 < len(data):
            if data[index] != 0xFF:
                index += 1
                continue
            marker = data[index + 1]
            index += 2
            if marker in {0xD8, 0xD9}:
                continue
            if index + 2 > len(data):
                return None
            segment_length = struct.unpack(">H", data[index : index + 2])[0]
            if segment_length < 2 or index + segment_length > len(data):
                return None
            if marker in {
                0xC0,
                0xC1,
                0xC2,
                0xC3,
                0xC5,
                0xC6,
                0xC7,
                0xC9,
                0xCA,
                0xCB,
                0xCD,
                0xCE,
                0xCF,
            }:
                if segment_length >= 7:
                    height, width = struct.unpack(">HH", data[index + 3 : index + 7])
                    return (width, height) if width > 0 and height > 0 else None
                return None
            index += segment_length
    return None


def _image_token_estimate_from_url(url: str) -> int:
    data = _data_url_bytes(url)
    if data is None:
        return _IMAGE_TOKEN_ESTIMATE
    dimensions = _image_dimensions_from_bytes(data)
    if dimensions is None:
        return _IMAGE_TOKEN_ESTIMATE
    width, height = dimensions
    return max(_IMAGE_TOKEN_ESTIMATE, (width * height + 749) // 750)


def _image_token_estimate_from_part(content: dict[str, Any]) -> int:
    image_url = content.get("image_url")
    url = image_url.get("url") if isinstance(image_url, dict) else image_url
    if not isinstance(url, str) or not url:
        source = content.get("source") if isinstance(content.get("source"), dict) else {}
        if source.get("type") == "base64":
            media_type = str(source.get("media_type") or "image/png")
            data = str(source.get("data") or "")
            url = f"data:{media_type};base64,{data}" if data else ""
        else:
            url = str(content.get("url") or "")
    if not url:
        return _IMAGE_TOKEN_ESTIMATE
    return _image_token_estimate_from_url(url)


def _estimate_image_content_tokens(content: Any) -> int:
    if isinstance(content, list):
        return sum(_estimate_image_content_tokens(item) for item in content)
    if isinstance(content, dict):
        item_type = content.get("type")
        count = _image_token_estimate_from_part(content) if item_type in {"image", "image_url", "input_image"} else 0
        if "content" in content:
            count += _estimate_image_content_tokens(content.get("content"))
        return count
    return 0


def _system_prompt_fingerprint(prompt: str) -> str:
    """Stable id for the raw system prompt a compression was derived from."""
    return hashlib.sha256(prompt.encode("utf-8", errors="replace")).hexdigest()


def _token_estimate_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                if item is not None:
                    parts.append(str(item))
                continue
            item_type = item.get("type")
            if item_type in {"text", "input_text", "output_text"} and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif item_type in {"image", "image_url"}:
                parts.append("[image]")
            elif "content" in item:
                parts.append(_token_estimate_text(item.get("content")))
        return " ".join(part for part in parts if part)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False, sort_keys=True)
    return str(content) if content is not None else ""


_TOKEN_ESTIMATOR: Any = None
_TOKEN_ESTIMATOR_TRIED = False
# A full conversation can be megabytes; encoding all of it on every truncation
# probe is wasteful, so long strings fall back to the cheap heuristic.
_TOKENIZER_MAX_CHARS = 200_000


def _count_tokens_with_tiktoken(text: str) -> int | None:
    """Count tokens with tiktoken, or return None when unavailable.

    Loading the BPE table costs time and may need a network fetch, so it is
    attempted once per process and every failure path degrades to the
    character heuristic instead of breaking truncation.
    """
    global _TOKEN_ESTIMATOR, _TOKEN_ESTIMATOR_TRIED
    if len(text) > _TOKENIZER_MAX_CHARS:
        return None
    if _TOKEN_ESTIMATOR is None and not _TOKEN_ESTIMATOR_TRIED:
        _TOKEN_ESTIMATOR_TRIED = True
        try:
            import tiktoken

            _TOKEN_ESTIMATOR = tiktoken.get_encoding("o200k_base")
        except Exception as e:  # pragma: no cover - depends on local install
            logger.info("[OpenClaw] tiktoken unavailable for token estimation (%s); using heuristic", e)
            _TOKEN_ESTIMATOR = None
    if _TOKEN_ESTIMATOR is None:
        return None
    try:
        return len(_TOKEN_ESTIMATOR.encode(text, disallowed_special=()))
    except Exception:
        return None


def _estimate_text_tokens(text: str) -> int:
    """Estimate tokens for a text blob.

    ``len(text) / 4`` assumes ASCII, where it errs on the safe side. It badly
    under-counts CJK and emoji (measured 0.44x and 0.25x of the real o200k
    count), which made truncation stop too early and forward an over-limit
    prompt. Prefer a real tokenizer and keep the heuristic as the fallback.
    """
    if not text:
        return 0
    counted = _count_tokens_with_tiktoken(text)
    if counted is not None:
        return counted
    return max(1, (len(text) + 3) // 4)


def _estimate_openai_body_input_tokens(openai_body: dict[str, Any]) -> int:
    """Return a provider-agnostic estimate of the prompt's input tokens.

    tiktoken is already a runtime dependency and is used here when available;
    the character heuristic remains the fallback so truncation stays active in
    any environment where the BPE table cannot be loaded.
    """
    messages = list(openai_body.get("messages") or [])
    tools = openai_body.get("tools")
    image_tokens = sum(_estimate_image_content_tokens(msg.get("content")) for msg in messages if isinstance(msg, dict))
    text_parts = []
    for msg in messages:
        if not isinstance(msg, dict):
            # A client may POST a bare string as a message; it still costs tokens.
            text_parts.append(str(msg))
            continue
        text_parts.append(f"{msg.get('role', '')}: {_token_estimate_text(msg.get('content'))}")
        if msg.get("tool_calls"):
            text_parts.append(json.dumps(msg.get("tool_calls"), ensure_ascii=False, sort_keys=True))
    if tools:
        text_parts.append(json.dumps(tools, ensure_ascii=False, sort_keys=True))
    text = "\n".join(part for part in text_parts if part)
    return max(1, _estimate_text_tokens(text) + image_tokens)


def _message_identity(message: dict[str, Any]) -> str:
    try:
        return json.dumps(message, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        return str(message)


def _split_leading_system_messages(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    index = 0
    while index < len(messages):
        msg = messages[index]
        if not isinstance(msg, dict) or msg.get("role") != "system":
            break
        index += 1
    return messages[:index], messages[index:]


def _canonical_overlap_message(message: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(message)
    if "content" in normalized:
        normalized["content"] = _flatten_message_content(normalized.get("content"))
    return normalized


def _merge_assistant_overlap_run(messages: list[dict[str, Any]]) -> dict[str, Any]:
    content_parts: list[str] = []
    tool_calls: list[Any] = []
    for msg in messages:
        content = _flatten_message_content(msg.get("content"))
        if content:
            content_parts.append(content)
        msg_tool_calls = msg.get("tool_calls")
        if isinstance(msg_tool_calls, list):
            tool_calls.extend(msg_tool_calls)

    merged: dict[str, Any] = {"role": "assistant", "content": " ".join(content_parts)}
    if tool_calls:
        merged["tool_calls"] = tool_calls
    return merged


def _messages_for_overlap(messages: list[dict[str, Any]]) -> list[tuple[str, int]]:
    entries: list[tuple[str, int]] = []
    index = 0
    while index < len(messages):
        msg = messages[index]
        if not isinstance(msg, dict):
            entries.append((_message_identity({"value": msg}), index + 1))
            index += 1
            continue

        if msg.get("role") == "assistant":
            run = [msg]
            next_index = index + 1
            has_tool_calls = isinstance(msg.get("tool_calls"), list) and bool(msg.get("tool_calls"))
            while next_index < len(messages):
                next_msg = messages[next_index]
                if not isinstance(next_msg, dict) or next_msg.get("role") != "assistant":
                    break
                run.append(next_msg)
                has_tool_calls = has_tool_calls or (
                    isinstance(next_msg.get("tool_calls"), list) and bool(next_msg.get("tool_calls"))
                )
                next_index += 1
            if has_tool_calls:
                entries.append((_message_identity(_merge_assistant_overlap_run(run)), next_index))
                index = next_index
                continue

        entries.append((_message_identity(_canonical_overlap_message(msg)), index + 1))
        index += 1
    return entries


def _merge_previous_response_messages(
    previous_messages: list[dict[str, Any]],
    current_messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not previous_messages:
        return current_messages
    if not current_messages:
        return previous_messages

    current_system_messages, current_body_messages = _split_leading_system_messages(current_messages)
    if current_system_messages:
        _, previous_body_messages = _split_leading_system_messages(previous_messages)
    else:
        previous_body_messages = previous_messages

    previous_entries = _messages_for_overlap(previous_body_messages)
    current_entries = _messages_for_overlap(current_body_messages)
    previous_keys = [key for key, _ in previous_entries]
    current_keys = [key for key, _ in current_entries]
    if current_keys[: len(previous_keys)] == previous_keys:
        return current_system_messages + current_body_messages

    max_overlap = min(len(previous_keys), len(current_keys))
    overlap = 0
    for size in range(max_overlap, 0, -1):
        if previous_keys[-size:] == current_keys[:size]:
            overlap = size
            break
    current_drop_index = current_entries[overlap - 1][1] if overlap else 0
    return current_system_messages + previous_body_messages + current_body_messages[current_drop_index:]


def _normalize_responses_content(content: Any) -> str:
    return responses_protocol.normalize_content_to_text(content)


def _responses_tools_to_openai_tools(tools: Any) -> list[dict]:
    return responses_protocol.tools_to_openai_tools(tools)


def _responses_to_openai_body(body: dict[str, Any], default_model: str) -> dict[str, Any]:
    try:
        return responses_protocol.to_openai_body(body, default_model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _responses_function_item_id(call_id: str, index: int) -> str:
    return responses_protocol.function_item_id(call_id, index)


def _openai_chat_to_responses_payload(payload: dict[str, Any], model: str) -> dict[str, Any]:
    return responses_protocol.from_openai_chat_payload(payload, model)


def _openai_to_anthropic_response(
    openai_resp: dict[str, Any],
    model: str,
    tool_names: set[str] | None = None,
) -> dict[str, Any]:
    return anthropic_protocol.from_openai_response(openai_resp, model, tool_names)


# ------------------------------------------------------------------ #
# SkillClawAPIServer                                                    #
# ------------------------------------------------------------------ #


def _message_tool_call_ids(message: dict) -> set[str]:
    """Collect the tool_call ids a message references (issued or answered)."""
    ids: set[str] = set()

    if not isinstance(message, dict):
        return ids

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if isinstance(call, dict) and call.get("id"):
                ids.add(str(call["id"]))

    if message.get("role") == "tool":
        call_id = message.get("tool_call_id")
        if call_id:
            ids.add(str(call_id))

    return ids


def _message_drop_units(messages: list[dict]) -> list[list[int]]:
    """Group message indices into units that must be dropped together.

    An assistant tool-call message and the role:tool results answering it are
    one unit: keeping a result without the turn that issued its ``tool_call_id``
    produces an orphan tool message, which OpenAI-compatible upstreams reject
    with ``400 invalid_request_error``. Every other message is its own unit.

    The Responses bridge emits one assistant message per ``function_call`` item,
    so parallel tool calls arrive as consecutive assistant siblings followed by
    their results. Those siblings and all their results form a single unit:
    grouping only an assistant with *following* results would let truncation
    drop one sibling and orphan the other's result.
    """
    units: list[list[int]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if not isinstance(message, dict) or not _message_tool_call_ids(message):
            units.append([index])
            index += 1
            continue

        # Absorb every consecutive assistant message that also issues tool
        # calls: they belong to the same tool round.
        group = [index]
        group_ids = set(_message_tool_call_ids(message))
        index += 1
        while (
            index < len(messages)
            and isinstance(messages[index], dict)
            and messages[index].get("role") == "assistant"
            and _message_tool_call_ids(messages[index])
        ):
            group_ids |= _message_tool_call_ids(messages[index])
            group.append(index)
            index += 1
        # Then absorb the results answering any id in the group.
        while index < len(messages) and _message_tool_call_ids(messages[index]) & group_ids:
            group_ids |= _message_tool_call_ids(messages[index])
            group.append(index)
            index += 1
        units.append(group)
    return units


def _merge_responses_item(payload: dict[str, Any], item: dict[str, Any]) -> None:
    """Add an output item to a Responses payload that omitted it."""
    output = payload.setdefault("output", [])
    if not isinstance(output, list):
        payload["output"] = output = []
    key = str(item.get("id") or item.get("call_id") or "")
    for existing in output:
        if isinstance(existing, dict) and key and str(existing.get("id") or existing.get("call_id") or "") == key:
            return
    output.append(item)


def _responses_finish_reason(payload: dict[str, Any]) -> str:
    """Map a Responses terminal status onto a chat ``finish_reason``."""
    if str(payload.get("status") or "") == "incomplete":
        details = payload.get("incomplete_details")
        reason = str(details.get("reason") or "") if isinstance(details, dict) else ""
        return "length" if reason == "max_output_tokens" else "stop"
    return "stop"


def _chat_tools_to_responses(tools: list[Any]) -> list[dict[str, Any]]:
    """Translate chat ``tools`` into Responses ``tools``.

    Both surfaces use a flat ``{"type": "function", ...}`` list, so the shape is
    preserved; only non-function tool types (which Responses does not accept
    the same way) are filtered out rather than forwarded blindly.
    """
    converted: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if str(tool.get("type") or "function") != "function":
            continue
        converted.append(tool)
    return converted


def _tui_client_discriminator(client_key: str) -> str:
    """Derive a short, stable id identifying the calling client.

    Clients that omit X-Session-Id used to share one pseudo-session per model,
    so two agents' turns interleaved. Hashing whatever identity the request
    carries (auth header, user-agent, remote peer) keeps them apart without
    trusting the value as a path or key.
    """
    raw = (client_key or "").strip()
    if not raw:
        return ""
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:12]


def _responses_item_user_turn(item: Any) -> bool:
    """Whether a Responses input item is a user turn."""
    if not isinstance(item, dict):
        return False
    if item.get("type") in {None, "message"} and item.get("role") == "user":
        return True
    return False


def _responses_item_group_key(item: Any) -> str:
    """Group key tying a function_call_output back to its function_call."""
    if not isinstance(item, dict):
        return ""
    if item.get("type") in {"function_call", "function_call_output", "custom_tool_call", "custom_tool_call_output"}:
        return str(item.get("call_id") or item.get("id") or "")
    return ""


def _responses_input_drop_units(items: list[Any]) -> list[list[int]]:
    """Group Responses input indices into units that drop together.

    A ``function_call`` and the ``function_call_output`` answering it are one
    unit: a Responses upstream rejects an output with no matching call, and a
    call with no output, both with 400 invalid_request_error.
    """
    units: list[list[int]] = []
    pending: dict[str, list[int]] = {}
    for index, item in enumerate(items):
        key = _responses_item_group_key(item)
        if not key:
            units.append([index])
            continue
        if isinstance(item, dict) and item.get("type") in {"function_call_output", "custom_tool_call_output"}:
            issued = pending.pop(key, None)
            if issued:
                issued.append(index)
                units.append(issued)
                continue
            units.append([index])
            continue
        if key in pending:
            pending[key].append(index)
        else:
            pending[key] = [index]
            units.append(pending[key])
    for orphan in pending.values():
        # An unanswered call is still a unit of its own; already in units.
        if not any(unit and unit[-1] == orphan[-1] for unit in units):
            units.append(list(orphan))
    return units


def _drop_oldest_response_units(items: list[Any], units: list[list[int]], count: int) -> list[Any]:
    dropped: set[int] = set()
    for unit in units[:count]:
        dropped.update(unit)
    return [item for i, item in enumerate(items) if i not in dropped]


def _truncate_responses_input(items: list[Any], max_prompt_tokens: int) -> list[Any]:
    """Trim a native Responses ``input`` list to fit the context budget.

    The native path had no truncation at all, so a long Codex conversation was
    forwarded verbatim and came back as an upstream 400. This mirrors the chat
    path: keep the newest item, drop whole tool groups oldest-first, and never
    settle on a cut with no user turn.
    """
    if max_prompt_tokens <= 0 or not isinstance(items, list) or not items:
        return items

    def _len(candidate: list[Any]) -> int:
        return _estimate_text_tokens(_token_estimate_text(candidate))

    original = _len(items)
    if original <= max_prompt_tokens:
        return items

    # A bare string input is a single user turn; leave it untouched.
    if any(not isinstance(item, dict) for item in items):
        return items

    units = _responses_input_drop_units(items)
    dropped = 0
    for step in range(1, len(units)):
        candidate = _drop_oldest_response_units(items, units, step)
        if not candidate:
            break
        dropped = step
        if any(_responses_item_user_turn(item) for item in candidate) and _len(candidate) <= max_prompt_tokens:
            break

    result = _drop_oldest_response_units(items, units, dropped)
    if not any(_responses_item_user_turn(item) for item in result):
        for step in range(len(units), dropped, -1):
            candidate = _drop_oldest_response_units(items, units, step)
            if candidate and any(_responses_item_user_turn(item) for item in candidate):
                dropped = step
                result = candidate
                break
    result_tokens = _len(result)
    if dropped:
        logger.info(
            "[OpenClaw] Responses input truncated: dropped %d oldest items (%d -> %d est tokens, limit=%d)",
            dropped,
            original,
            result_tokens,
            max_prompt_tokens,
        )
    return result


def _has_user_turn(messages: list[Any]) -> bool:
    """Whether a message list still carries a user turn.

    OpenAI-compatible upstreams reject a conversation with no user turn, so
    truncation must never settle on a cut that removes all of them.
    """
    return any(isinstance(m, dict) and m.get("role") == "user" for m in messages)


def _drop_oldest_units(messages: list[dict], units: list[list[int]], count: int) -> list[dict]:
    """Drop the ``count`` oldest units, keeping the rest in order."""
    dropped: set[int] = set()
    for unit in units[:count]:
        dropped.update(unit)
    return [m for i, m in enumerate(messages) if i not in dropped]


def _sanitize_forward_messages(messages: list[Any]) -> list[Any]:
    """Make a message list acceptable to OpenAI-compatible upstreams.

    This is the single choke point for every inbound protocol (chat, Anthropic
    Messages, Responses), so the three shapes that upstreams reject with a
    deterministic ``400 invalid_request_error`` are fixed once:

    1. a ``role:tool`` result whose issuing assistant turn is absent, or whose
       ``tool_call_id`` is empty (an Anthropic ``tool_result`` block with no
       ``tool_use_id``);
    2. a conversation that ends on an unanswered assistant ``tool_calls``;
    3. a conversation with no ``user`` turn at all.

    Fixing them here rather than per-protocol matters because the rejected
    request was previously retried as if it were a transient failure, turning a
    client bug into ~33s of latency and a 502.
    """
    if not isinstance(messages, list) or not messages:
        return messages

    # 1. Drop tool results with no id or no issuer anywhere in the request.
    issued: set[str] = set()
    for message in messages:
        if isinstance(message, dict):
            issued |= _message_tool_call_ids(message)

    kept: list[Any] = []
    dropped_orphans = 0
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "tool":
            call_id = str(message.get("tool_call_id") or "").strip()
            if not call_id or call_id not in issued:
                dropped_orphans += 1
                continue
        kept.append(message)
    if dropped_orphans:
        logger.info("[OpenClaw] dropped %d orphan tool result(s) before forwarding", dropped_orphans)

    # A tool result must follow the assistant turn that issued it. Reorder
    # defensively: an upstream reads positionally, not by id lookup.
    kept = _hoist_tool_results_after_assistant(kept)

    # 2. An unanswered trailing tool_call is rejected. Synthesize the missing
    #    result rather than dropping the call, so the client still sees the
    #    tool the model asked for and can answer it next turn.
    if kept and isinstance(kept[-1], dict) and _assistant_message_has_tool_calls(kept[-1]):
        pending = [
            str(call.get("id") or "")
            for call in (kept[-1].get("tool_calls") or [])
            if isinstance(call, dict) and call.get("id")
        ]
        answered = {
            str(m.get("tool_call_id") or "")
            for m in kept
            if isinstance(m, dict) and m.get("role") == "tool"
        }
        for call_id in pending:
            if call_id in answered:
                continue
            kept.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": "[tool result unavailable: client ended the turn without calling this tool]",
                }
            )

    # 3. Guarantee a user turn so the request is never conversation-less.
    if not _has_user_turn(kept):
        logger.info("[OpenClaw] injected a placeholder user turn: request had none")
        kept.append({"role": "user", "content": "Continue."})
    return kept


def _hoist_tool_results_after_assistant(messages: list[Any]) -> list[Any]:
    """Move each ``role:tool`` message directly after its issuing assistant."""
    out: list[Any] = []
    pending: dict[str, list[Any]] = {}
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "tool":
            call_id = str(message.get("tool_call_id") or "").strip()
            pending.setdefault(call_id, []).append(message)
            continue
        out.append(message)
        if isinstance(message, dict) and message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                call_id = str(call.get("id") or "").strip()
                waiting = pending.pop(call_id, None)
                if waiting:
                    out.extend(waiting)
    # Anything still waiting had no issuer; drop it rather than reorder blindly.
    return out


def _repair_orphan_tool_results(messages: list[dict]) -> list[dict]:
    """Drop ``role:tool`` results whose issuing assistant turn is gone.

    Truncation normally keeps tool rounds whole, but a client can hand us a
    result whose assistant turn is not in the request at all, and truncation
    can split a round when a result is separated from its issuer by a user
    turn. Either way the forwarded body would contain an orphan tool message,
    which OpenAI-compatible upstreams reject with 400 invalid_request_error.
    """
    issued: set[str] = set()
    for message in messages:
        if isinstance(message, dict):
            issued |= _message_tool_call_ids(message)
    repaired: list[dict] = []
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "tool":
            call_id = str(message.get("tool_call_id") or "")
            if call_id and call_id not in issued:
                logger.info("[OpenClaw] dropping orphan tool result %s during truncation", call_id)
                continue
        repaired.append(message)
    return repaired


class SkillClawAPIServer:
    """Proxy between client agents and the upstream model with SkillClaw hooks.

    OpenClaw sends ``X-Session-Id`` and ``X-Turn-Type`` headers with every
    request. The proxy injects skills, records conversation artifacts when
    enabled, and can attach PRM scoring when configured. Side tasks
    (``turn_type != "main"``) are forwarded but do not generate the main
    conversation artifact path.

    Parameters
    ----------
    config:
        SkillClawConfig instance.
    skill_manager:
        Optional SkillManager for injecting skills into system prompts.
    prm_scorer:
        Optional PRMScorer for turn feedback.
    """

    def __init__(
        self,
        config: SkillClawConfig,
        sampling_client=None,
        skill_manager: Optional[SkillManager] = None,
        prm_scorer: Optional[PRMScorer] = None,
        last_request_tracker=None,
    ):
        self.config = config
        self._sampling_client = sampling_client
        self.skill_manager = skill_manager
        self.prm_scorer = prm_scorer
        self._last_request_tracker = last_request_tracker
        self._last_request_at = time.time()

        self._served_model = config.served_model_name
        self._expected_api_key = config.proxy_api_key
        # System prompt compression is only used for OpenClaw (whose verbose
        # system prompt benefits from compression).  Non-OpenClaw agents send
        # short/no system prompts, and the compressed OpenClaw text can trigger
        # content filters on strict providers (e.g. Azure).
        self._compress_system_prompt = config.claw_type == "openclaw"
        cache_suffix = f"{config.claw_type}_{config.llm_provider}"
        self._system_prompt_cache_file = os.path.join(config.record_dir, f"system_prompt_cache_{cache_suffix}.json")

        # State machines
        self._turn_counts: dict[str, int] = {}
        self._user_turn_counts: dict[str, int] = {}
        self._pending_turn_data: dict[str, dict[int, dict]] = {}  # session → {turn → data}
        self._prm_tasks: dict[str, dict[int, asyncio.Task]] = {}  # session → {turn → task}
        self._pending_records: dict[str, dict] = {}  # for record logging
        self._session_scored_turns: dict[str, int] = {}  # session -> finalized PRM turn count
        self._session_turns: dict[str, list] = {}
        self._session_last_active: dict[str, float] = {}  # session -> unix_ts
        self._closing_sessions: set[str] = set()  # session ids currently being closed
        self._background_tasks: set[asyncio.Task] = set()  # transient async tasks (upload, submit)
        self._responses_store: dict[str, dict[str, Any]] = {}  # response_id -> stored response/history
        self._response_id_by_session: dict[str, set[str]] = {}  # session -> owned response ids
        self._closed_session_ids: set[str] = set()  # recently closed sessions, for store sweeping
        self._in_flight_sessions: set[str] = set()  # sessions with a request currently running
        self._session_sweeper_task: Optional[asyncio.Task] = None
        self._skill_reload_task: Optional[asyncio.Task] = None
        # Read from the dataclass directly: these fields now exist, so an
        # operator's config.yaml value is honoured instead of being discarded
        # in favour of a hard-coded default.
        self._session_idle_close_seconds = max(
            0,
            _coerce_int(getattr(config, "session_idle_close_seconds", None), _SESSION_IDLE_CLOSE_SECONDS),
        )
        self._session_sweep_interval_seconds = max(
            1,
            _coerce_int(getattr(config, "session_sweep_interval_seconds", None), _SESSION_SWEEP_INTERVAL_SECONDS),
        )
        self._shutdown_drain_timeout_seconds = max(
            1,
            _coerce_int(getattr(config, "shutdown_drain_timeout_seconds", None), _SHUTDOWN_DRAIN_TIMEOUT_SECONDS),
        )
        self._responses_store_max_entries = max(
            1,
            _coerce_int(getattr(config, "responses_store_max_entries", None), _RESPONSES_STORE_MAX_ENTRIES),
        )
        self._responses_store_ttl_seconds = max(
            1.0,
            float(getattr(config, "responses_store_ttl_seconds", None) or _RESPONSES_STORE_TTL_SECONDS),
        )
        self._skill_reload_interval_seconds = max(
            5,
            int(getattr(config, "sharing_skill_reload_interval_seconds", 30) or 30),
        )

        # Session boundary detection for non-OpenClaw agents (QwenPaw, IronClaw, etc.)
        # Maps pseudo-session key (e.g. "tui-<client>-<model>") to tracking metadata.
        self._tui_session_meta: dict[str, dict] = {}
        # Stable per-client discriminator so two agents on the same model do not
        # collapse into one pseudo-session. Replaced by the first request's
        # client identity; falls back to a per-process constant.
        self._tui_client_id = os.environ.get("SKILLCLAW_CLIENT_ID", "").strip() or f"p{os.getpid()}x"
        _INACTIVITY_TIMEOUT = 300  # seconds — treat as new session after 5 min idle
        self._tui_inactivity_timeout = _INACTIVITY_TIMEOUT

        # Record files
        self._record_file = ""
        self._prm_record_file = ""
        if config.record_enabled:
            os.makedirs(config.record_dir, exist_ok=True)
            self._record_file = os.path.join(config.record_dir, "conversations.jsonl")
            self._prm_record_file = os.path.join(config.record_dir, "prm_scores.jsonl")
            with open(self._record_file, "a"):
                pass
            with open(self._prm_record_file, "a"):
                pass

        self.app = self._build_app()

        # Threading lifecycle (set by start())
        self._server: Optional[uvicorn.Server] = None
        self._thread: Optional[threading.Thread] = None
        self._ready_event = threading.Event()
        self._server_stopped_event = threading.Event()

    # ------------------------------------------------------------------ #
    # FastAPI app                                                          #
    # ------------------------------------------------------------------ #

    def _build_app(self) -> FastAPI:
        owner = self

        @asynccontextmanager
        async def lifespan(_app: FastAPI):
            owner._ready_event.set()
            owner._start_session_idle_sweeper()
            owner._start_skill_reload_polling()
            try:
                yield
            finally:
                owner._ready_event.clear()
                await owner._shutdown_cleanup()

        app = FastAPI(title="SkillClaw Proxy", lifespan=lifespan)
        app.state.owner = self

        @app.get("/healthz")
        async def healthz():
            return {"ok": True}

        @app.post("/internal/reload-skills")
        async def reload_skills(
            request: Request,
            authorization: Optional[str] = Header(default=None),
        ):
            owner: SkillClawAPIServer = request.app.state.owner
            await owner._check_auth(authorization)
            await owner._pull_skills_from_cloud()
            skill_count = len(owner.skill_manager.get_all_skills()) if owner.skill_manager else 0
            return {"ok": True, "skills": skill_count}

        @app.get("/v1/models")
        async def list_models(
            request: Request,
            authorization: Optional[str] = Header(default=None),
        ):
            owner: SkillClawAPIServer = request.app.state.owner
            await owner._check_auth(authorization)
            model_id = owner._served_model
            return JSONResponse(
                content={
                    "object": "list",
                    "data": [
                        {
                            "id": model_id,
                            "object": "model",
                            "created": 0,
                            "owned_by": "skillclaw",
                        }
                    ],
                }
            )

        @app.post("/v1/chat/completions")
        async def chat_completions(
            request: Request,
            authorization: Optional[str] = Header(default=None),
            x_session_id: Optional[str] = Header(default=None),
            x_turn_type: Optional[str] = Header(default=None),
            x_session_done: Optional[str] = Header(default=None),
        ):
            owner: SkillClawAPIServer = request.app.state.owner
            # Update idle tracker so the scheduler knows the user is active
            owner._mark_request_activity()
            await owner._check_auth(authorization)

            body = await request.json()
            incoming_messages = body.get("messages", [])
            if isinstance(incoming_messages, list):
                rewritten_messages, _ = _rewrite_new_session_bootstrap_prompt(incoming_messages)
                body["messages"] = rewritten_messages
            _raw_sid = x_session_id or body.get("session_id") or ""
            # OpenClaw sends X-Session-Id/X-Turn-Type on every request.
            # Non-OpenClaw agents (QwenPaw, IronClaw, etc.) don't — detect
            # session boundaries heuristically so session upload and state
            # cleanup still work correctly.
            if _raw_sid:
                session_id = _raw_sid
                turn_type = _resolve_turn_type(x_turn_type, body.get("turn_type"), default="main")
            else:
                msg_count = len(body.get("messages") or [])
                session_id = await owner._resolve_tui_session(
                    body.get("model", "default"),
                    msg_count,
                    owner._tui_client_key(request, authorization),
                )
                turn_type = _resolve_turn_type(x_turn_type, body.get("turn_type"), default="main")
            session_done = _resolve_session_done(x_session_done, body.get("session_done"))
            # Do not infer session_done from bootstrap text — only explicit
            # X-Session-Done or body session_done trigger session close.

            stream = bool(body.get("stream", False))
            result = await owner._handle_request(
                body,
                session_id=session_id,
                turn_type=turn_type,
                session_done=session_done,
            )
            if stream:
                return StreamingResponse(owner._stream_response(result), media_type="text/event-stream")
            return JSONResponse(content=result["response"])

        @app.post("/v1/responses")
        async def responses(
            request: Request,
            authorization: Optional[str] = Header(default=None),
            x_session_id: Optional[str] = Header(default=None),
            codex_session_id: Optional[str] = Header(default=None, alias="session_id"),
            x_turn_type: Optional[str] = Header(default=None),
            x_session_done: Optional[str] = Header(default=None),
        ):
            owner: SkillClawAPIServer = request.app.state.owner
            owner._mark_request_activity()
            await owner._check_auth(authorization)

            body = await request.json()
            if owner._responses_native_enabled():
                record_body = copy.deepcopy(body)
                turn_type = _resolve_turn_type(x_turn_type, body.get("turn_type"), default="main")
                injected_skills = owner._prepare_native_responses_body_inplace(body, turn_type=turn_type)
                _raw_sid = x_session_id or codex_session_id or body.get("session_id") or ""
                session_id = _raw_sid or await owner._resolve_tui_session(
                    body.get("model", owner._served_model),
                    len(body.get("input", []) if isinstance(body.get("input"), list) else []),
                )
                session_done = _resolve_session_done(x_session_done, body.get("session_done"))
                if bool(body.get("stream", False)):
                    return StreamingResponse(
                        owner._stream_and_track_responses(
                            body,
                            record_body=record_body,
                            session_id=session_id,
                            turn_type=turn_type,
                            injected_skills=injected_skills,
                            session_done=session_done,
                        ),
                        media_type="text/event-stream",
                    )
                response_payload = await owner._forward_to_llm_responses(body)
                owner._record_responses_turn(
                    session_id,
                    record_body,
                    response_payload,
                    turn_type=turn_type,
                    injected_skills=injected_skills,
                    session_done=session_done,
                )
                return JSONResponse(content=response_payload)

            previous_response_id = str(body.get("previous_response_id") or "").strip()
            store_response = bool(body.get("store", True))
            _raw_sid = x_session_id or codex_session_id or body.get("session_id") or ""
            if _raw_sid:
                session_id = _raw_sid
                turn_type = _resolve_turn_type(x_turn_type, body.get("turn_type"), default="main")
            else:
                msg_count = len(body.get("input", []) if isinstance(body.get("input"), list) else [])
                session_id = await owner._resolve_tui_session(
                    body.get("model", owner._served_model),
                    msg_count,
                    owner._tui_client_key(request, authorization),
                )
                turn_type = _resolve_turn_type(x_turn_type, body.get("turn_type"), default="main")

            openai_body = _responses_to_openai_body(body, owner._served_model)
            openai_body["_skillclaw_protocol"] = _PROTOCOL_RESPONSES_COMPAT
            if previous_response_id:
                stored = owner._get_stored_response(previous_response_id, session_id)
                openai_body["messages"] = _merge_previous_response_messages(
                    list(stored.get("messages") or []),
                    list(openai_body.get("messages") or []),
                )
            session_done = _resolve_session_done(x_session_done, body.get("session_done"))

            result = await owner._handle_request(
                openai_body,
                session_id=session_id,
                turn_type=turn_type,
                session_done=session_done,
            )
            response_payload = _openai_chat_to_responses_payload(
                result["response"],
                model=openai_body.get("model", owner._served_model),
            )
            # The proxy id is derived from the upstream chat id, which can repeat
            # across turns. Two turns then collided on one key and the second
            # write clobbered the first, so a client resuming turn 1 by id got
            # turn 2's history.
            response_payload = {
                **response_payload,
                "id": owner._unique_response_id(response_payload.get("id")),
            }
            choices = result.get("response", {}).get("choices") or [{}]
            first_choice = choices[0] if isinstance(choices[0], dict) else {}
            assistant_message = first_choice.get("message") or {}
            if not isinstance(assistant_message, dict) or not assistant_message.get("role"):
                # Storing a roleless fragment made the next turn replay a
                # message that no upstream would accept.
                assistant_message = {"role": "assistant", "content": ""}
            if store_response:
                # Store what was actually forwarded, not the pre-truncation
                # list: replaying the full history re-inflated a prompt the
                # proxy had just deliberately trimmed.
                owner._store_response(
                    response_payload,
                    messages=[
                        *list(result.get("forwarded_messages") or openai_body.get("messages") or []),
                        assistant_message,
                    ],
                    session_id=session_id,
                )
            if bool(body.get("stream", False)):
                return StreamingResponse(
                    owner._stream_responses_response(response_payload),
                    media_type="text/event-stream",
                )
            return JSONResponse(content=response_payload)

        @app.get("/v1/responses/{response_id}")
        async def get_response(
            response_id: str,
            request: Request,
            authorization: str = Header(default=None),
            x_session_id: Optional[str] = Header(default=None),
        ):
            owner: SkillClawAPIServer = request.app.state.owner
            await owner._check_auth(authorization)
            stored = owner._get_stored_response(response_id, x_session_id)
            return JSONResponse(content=stored["response"])

        @app.delete("/v1/responses/{response_id}")
        async def delete_response(
            response_id: str,
            request: Request,
            authorization: str = Header(default=None),
            x_session_id: Optional[str] = Header(default=None),
        ):
            owner: SkillClawAPIServer = request.app.state.owner
            await owner._check_auth(authorization)
            # Ownership is checked before the pop so an unauthorised id neither
            # reveals existence nor destroys another session's entry.
            owner._get_stored_response(response_id, x_session_id)
            owner._responses_store.pop(response_id, None)
            return JSONResponse(content={"id": response_id, "object": "response", "deleted": True})

        # ---------------------------------------------------------------- #
        # Anthropic-compatible endpoint — used by NanoClaw (credential proxy
        # forwards container Anthropic SDK calls to ANTHROPIC_BASE_URL).
        # ---------------------------------------------------------------- #

        @app.post("/v1/messages/count_tokens")
        async def anthropic_count_tokens(
            request: Request,
            authorization: Optional[str] = Header(default=None),
            x_api_key: Optional[str] = Header(default=None, alias="x-api-key"),
        ):
            owner: SkillClawAPIServer = request.app.state.owner
            owner._mark_request_activity()
            auth_header = authorization or (f"Bearer {x_api_key}" if x_api_key else None)
            await owner._check_auth(auth_header)

            raw_body = await request.json()
            openai_body = _anthropic_to_openai_body(raw_body)
            input_tokens = _estimate_openai_body_input_tokens(openai_body)
            return JSONResponse(content={"input_tokens": input_tokens})

        @app.post("/v1/messages")
        async def anthropic_messages(
            request: Request,
            authorization: Optional[str] = Header(default=None),
            x_api_key: Optional[str] = Header(default=None, alias="x-api-key"),
            x_session_id: Optional[str] = Header(default=None),
            x_claude_code_session_id: Optional[str] = Header(default=None, alias="x-claude-code-session-id"),
            x_turn_type: Optional[str] = Header(default=None),
            x_session_done: Optional[str] = Header(default=None),
        ):
            owner: SkillClawAPIServer = request.app.state.owner
            owner._mark_request_activity()
            # Accept Anthropic-style x-api-key as well as Bearer token.
            auth_header = authorization or (f"Bearer {x_api_key}" if x_api_key else None)
            await owner._check_auth(auth_header)

            raw_body = await request.json()
            stream = bool(raw_body.get("stream", False))
            tool_names = _anthropic_request_tool_names(raw_body)
            openai_body = _anthropic_to_openai_body(raw_body)
            openai_body["_skillclaw_protocol"] = _PROTOCOL_ANTHROPIC_MESSAGES
            model = raw_body.get("model") or owner._served_model

            incoming_messages = openai_body.get("messages", [])
            if isinstance(incoming_messages, list):
                rewritten_messages, _ = _rewrite_new_session_bootstrap_prompt(incoming_messages)
                openai_body["messages"] = rewritten_messages

            _raw_sid = x_session_id or x_claude_code_session_id or raw_body.get("session_id") or ""
            if _raw_sid:
                session_id = _raw_sid
                turn_type = _resolve_turn_type(x_turn_type, raw_body.get("turn_type"), default="main")
            else:
                msg_count = len(openai_body.get("messages") or [])
                session_id = await owner._resolve_tui_session(model, msg_count)
                turn_type = _resolve_turn_type(x_turn_type, raw_body.get("turn_type"), default="main")
            session_done = _resolve_session_done(x_session_done, raw_body.get("session_done"))

            result = await owner._handle_request(
                openai_body,
                session_id=session_id,
                turn_type=turn_type,
                session_done=session_done,
            )
            if stream:
                return StreamingResponse(
                    owner._stream_anthropic_response(result, model, tool_names),
                    media_type="text/event-stream",
                )
            return JSONResponse(content=_openai_to_anthropic_response(result["response"], model, tool_names))

        return app

    async def _check_auth(self, authorization: Optional[str]):
        if not self._expected_api_key:
            return
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="missing bearer token")
        token = authorization.split(" ", 1)[1].strip()
        if token != self._expected_api_key:
            raise HTTPException(status_code=401, detail="invalid api key")

    def _mark_request_activity(self) -> None:
        self._last_request_at = time.time()
        if self._last_request_tracker is not None:
            try:
                self._last_request_tracker.touch()
            except Exception:
                pass

    def last_request_age_seconds(self) -> Optional[float]:
        last = getattr(self, "_last_request_at", None)
        if last is None:
            return None
        return max(0.0, time.time() - float(last))

    def active_session_count(self) -> int:
        return len(self._collect_active_session_ids())

    def is_idle_for_validation(self, idle_after_seconds: int) -> bool:
        age = self.last_request_age_seconds()
        if age is None:
            return False
        if self.active_session_count() > 0:
            return False
        return age >= max(0, int(idle_after_seconds))

    # ------------------------------------------------------------------ #
    # Responses store (previous_response_id continuation)                #
    # ------------------------------------------------------------------ #

    def _unique_response_id(self, base_id: Any) -> str:
        """Return a store-unique id derived from the upstream response id.

        The payload id comes from the upstream chat id, which some providers
        reuse across turns. Keying the store on it made a second turn clobber
        the first, so a client resuming an earlier turn by id silently received
        the later turn's history.
        """
        base = str(base_id or "").strip() or "resp"
        candidate = base
        suffix = 1
        while candidate in self._responses_store:
            suffix += 1
            candidate = f"{base}-{suffix}"
        return candidate

    def _store_response(self, response_payload: dict[str, Any], *, messages: list, session_id: str) -> None:
        """Remember a response for ``previous_response_id`` continuation."""
        response_id = str(response_payload.get("id") or "")
        if not response_id:
            return
        self._prune_responses_store()
        self._responses_store[response_id] = {
            "response": response_payload,
            "messages": messages,
            "session_id": session_id,
            "stored_at": time.time(),
        }
        self._response_id_by_session.setdefault(session_id, set()).add(response_id)
        while len(self._responses_store) > self._responses_store_max_entries:
            oldest = min(
                self._responses_store.items(),
                key=lambda item: item[1].get("stored_at", 0.0),
            )[0]
            self._discard_response(oldest)

    def _discard_response(self, response_id: str) -> None:
        stored = self._responses_store.pop(response_id, None)
        if stored is None:
            return
        owner = stored.get("session_id")
        if owner:
            ids = self._response_id_by_session.get(owner)
            if ids:
                ids.discard(response_id)
                if not ids:
                    self._response_id_by_session.pop(owner, None)

    def _prune_responses_store(self) -> None:
        """Evict expired entries and every entry owned by a closed session."""
        now = time.time()
        expired = [
            response_id
            for response_id, stored in self._responses_store.items()
            if now - float(stored.get("stored_at", 0.0)) > self._responses_store_ttl_seconds
        ]
        for response_id in expired:
            self._discard_response(response_id)
        for session_id in self._closed_session_ids:
            for response_id in list(self._response_id_by_session.get(session_id, ())):
                self._discard_response(response_id)
        if len(self._responses_store) > self._responses_store_max_entries:
            excess = len(self._responses_store) - self._responses_store_max_entries
            for response_id in list(self._responses_store)[:excess]:
                self._discard_response(response_id)

    def _get_stored_response(self, response_id: str, session_id: Optional[str]) -> dict[str, Any]:
        """Return a stored response, enforcing session ownership.

        Without this check any client that learned (or guessed) a response id
        could resume another session's conversation and have its private turns
        forwarded upstream.
        """
        self._prune_responses_store()
        stored = self._responses_store.get(response_id)
        if stored is None:
            raise HTTPException(status_code=404, detail=f"response not found: {response_id}")
        owner = stored.get("session_id")
        if owner and session_id and owner != session_id:
            logger.warning(
                "[OpenClaw] refused cross-session responses lookup: id=%s owner=%s requester=%s",
                response_id,
                owner,
                session_id,
            )
            # Same 404 as an unknown id so existence is not disclosed.
            raise HTTPException(status_code=404, detail=f"response not found: {response_id}")
        return stored

    def _release_session_responses(self, session_id: str) -> None:
        """Drop every stored response owned by a closing session."""
        for response_id in list(self._response_id_by_session.get(session_id, ())):
            self._discard_response(response_id)
        self._response_id_by_session.pop(session_id, None)
        self._closed_session_ids.add(session_id)
        if len(self._closed_session_ids) > _CLOSED_SESSION_MEMORY:
            self._closed_session_ids.clear()

    # ------------------------------------------------------------------ #
    # TUI session boundary detection (QwenPaw / IronClaw / generic clients) #
    # ------------------------------------------------------------------ #

    async def _resolve_tui_session(self, model: str, msg_count: int, client_key: str = "") -> str:
        """Return a session_id for agents that don't send X-Session-Id.

        Detects new-conversation boundaries by two heuristics:
          1. Message count dropped — the client started a fresh conversation.
          2. Inactivity timeout — the user was idle for >N seconds.

        When a boundary is detected the old session is flushed (session data
        uploaded, state dicts cleaned up) and a new unique id is assigned.
        """
        import uuid

        # The key used to be `tui-{model}` only, so two clients on the same
        # model shared one pseudo-session: their turns interleaved in
        # _session_turns and one client's private content was uploaded under
        # the other's session id. Include the client identity when available,
        # and bound the map so an attacker cannot grow it with fake model names.
        client_id = _tui_client_discriminator(client_key) or self._tui_client_id
        tui_key = f"tui-{client_id}-{model}" if client_id else f"tui-anon-{model}"
        if len(self._tui_session_meta) > _TUI_SESSION_MAX_ENTRIES:
            self._tui_session_meta.clear()
        now = time.time()
        meta = self._tui_session_meta.get(tui_key)

        if meta is None:
            # First request for this model — start a fresh session.
            sid = f"tui-{model}-{uuid.uuid4().hex[:8]}"
            self._tui_session_meta[tui_key] = {
                "session_id": sid,
                "last_msg_count": msg_count,
                "last_request_time": now,
            }
            logger.info("[SessionDetect] new TUI session %s (first request)", sid)
            return sid

        new_session = False
        if msg_count < meta["last_msg_count"]:
            # Message count dropped → client started a new conversation.
            new_session = True
            logger.info(
                "[SessionDetect] msg count dropped %d → %d — new session",
                meta["last_msg_count"],
                msg_count,
            )
        elif (now - meta["last_request_time"]) > self._tui_inactivity_timeout:
            new_session = True
            idle_sec = int(now - meta["last_request_time"])
            logger.info(
                "[SessionDetect] inactivity %ds > %ds — new session",
                idle_sec,
                self._tui_inactivity_timeout,
            )

        if new_session:
            old_sid = meta["session_id"]
            await self._close_session(old_sid, reason="tui_boundary")
            sid = f"tui-{model}-{uuid.uuid4().hex[:8]}"
            self._tui_session_meta[tui_key] = {
                "session_id": sid,
                "last_msg_count": msg_count,
                "last_request_time": now,
            }
            logger.info("[SessionDetect] new TUI session %s (replacing %s)", sid, old_sid)
            return sid

        # Same session — update tracking.
        meta["last_msg_count"] = msg_count
        meta["last_request_time"] = now
        return meta["session_id"]

    @staticmethod
    def _tui_client_key(request: Request, authorization: str) -> str:
        """Build the per-client discriminator used for pseudo-session keys."""
        client_host = getattr(getattr(request, "client", None), "host", "") or ""
        user_agent = request.headers.get("user-agent", "")
        auth = (authorization or "").strip()
        # The auth token identifies the caller; host and user-agent separate
        # distinct local agents that share one credential.
        return "|".join([auth, client_host, user_agent])

    def _touch_session(self, session_id: str) -> None:
        if session_id:
            self._session_last_active[session_id] = time.time()

    def _session_has_pending_work(self, session_id: str) -> bool:
        """Whether a session still owns state the sweeper must not reclaim."""
        if session_id in self._in_flight_sessions:
            return True
        for mapping in (
            self._pending_turn_data,
            self._prm_tasks,
            self._pending_records,
            self._session_turns,
        ):
            value = mapping.get(session_id)
            if value:
                return True
        return bool(self._turn_counts.get(session_id))

    @contextmanager
    def _in_flight_request(self, session_id: str):
        """Mark a request as in-flight so the idle sweeper leaves it alone.

        A long request used to be reaped mid-flight, and its orphaned state was
        written afterwards without a fresh timestamp, so it could never be
        swept again.
        """
        self._in_flight_sessions.add(session_id)
        try:
            yield
        finally:
            self._in_flight_sessions.discard(session_id)
            # Re-arm the idle clock: the request just finished, so the session
            # must be collectable again once the new threshold elapses.
            self._touch_session(session_id)

    def _collect_active_session_ids(self) -> list[str]:
        # Only sessions with real pending work count. Unioning dict *keys* kept
        # a session active after every turn had been finalized, so idle
        # validation could never run on a proxy that had served any session.
        session_ids: set[str] = set()
        for mapping in (
            self._pending_turn_data,
            self._prm_tasks,
            self._pending_records,
            self._session_turns,
        ):
            session_ids.update(sid for sid, value in mapping.items() if value)
        session_ids.update(sid for sid, count in self._turn_counts.items() if count)
        session_ids.update(self._in_flight_sessions)
        return sorted(s for s in session_ids if s and s not in self._closing_sessions)

    def _collect_idle_session_ids(self, now: Optional[float] = None) -> list[str]:
        if self._session_idle_close_seconds <= 0:
            return []
        if now is None:
            now = time.time()
        threshold = float(self._session_idle_close_seconds)
        return sorted(
            sid
            for sid, ts in self._session_last_active.items()
            if sid
            and sid not in self._closing_sessions
            and (now - float(ts)) >= threshold
            # Never reap a request that is still running upstream.
            and sid not in self._in_flight_sessions
        )

    def _start_session_idle_sweeper(self) -> None:
        if self._session_idle_close_seconds <= 0:
            logger.info("[SessionDetect] idle sweeper disabled (timeout <= 0)")
            return
        if self._session_sweeper_task is not None and not self._session_sweeper_task.done():
            return
        self._session_sweeper_task = asyncio.create_task(self._session_idle_sweeper_loop())
        self._session_sweeper_task.add_done_callback(self._task_done_cb)
        logger.info(
            "[SessionDetect] idle sweeper started (timeout=%ss interval=%ss)",
            self._session_idle_close_seconds,
            self._session_sweep_interval_seconds,
        )

    async def _session_idle_sweeper_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._session_sweep_interval_seconds)
                stale_ids = self._collect_idle_session_ids()
                for sid in stale_ids:
                    await self._close_session(sid, reason="idle_timeout")
        except asyncio.CancelledError:
            logger.info("[SessionDetect] idle sweeper stopped")
            raise

    async def _await_background_tasks(self, timeout_seconds: float) -> None:
        pending = [t for t in list(self._background_tasks) if not t.done()]
        if not pending:
            return
        done, still_pending = await asyncio.wait(pending, timeout=timeout_seconds)
        if still_pending:
            logger.warning(
                "[OpenClaw] background drain timeout: %d task(s) still running",
                len(still_pending),
            )
            for task in still_pending:
                task.cancel()
            await asyncio.gather(*still_pending, return_exceptions=True)
        else:
            logger.info("[OpenClaw] background drain complete (%d task(s))", len(done))

    def _start_skill_reload_polling(self) -> None:
        if not self.config.sharing_enabled:
            return
        mode = str(getattr(self.config, "sharing_skill_reload_mode", "") or "poll").strip().lower()
        if mode != "poll":
            return
        if self._skill_reload_task is not None and not self._skill_reload_task.done():
            return
        self._skill_reload_task = asyncio.create_task(self._skill_reload_poll_loop())
        self._skill_reload_task.add_done_callback(self._task_done_cb)
        logger.info(
            "[SkillHub] skill reload polling enabled interval=%ds",
            self._skill_reload_interval_seconds,
        )

    async def _skill_reload_poll_loop(self) -> None:
        consecutive_failures = 0
        first_pull = True
        try:
            while True:
                if first_pull:
                    first_pull = False
                else:
                    jitter = random.uniform(0, self._skill_reload_interval_seconds * 0.1)
                    backoff = min(consecutive_failures * 5.0, 60.0)
                    await asyncio.sleep(self._skill_reload_interval_seconds + jitter + backoff)
                try:
                    await self._pull_skills_from_cloud()
                    consecutive_failures = 0
                except Exception as exc:
                    consecutive_failures += 1
                    logger.warning(
                        "[SkillHub] skill reload poll failed (streak=%d): %s",
                        consecutive_failures,
                        exc,
                    )
        except asyncio.CancelledError:
            logger.info("[SkillHub] skill reload polling stopped")
            raise

    async def _drain_active_sessions(self, reason: str) -> None:
        active_ids = self._collect_active_session_ids()
        if not active_ids:
            return
        logger.info("[SessionDetect] draining %d active session(s): reason=%s", len(active_ids), reason)
        for sid in active_ids:
            await self._close_session(sid, reason=reason)

    async def _shutdown_cleanup(self) -> None:
        if self._skill_reload_task is not None:
            self._skill_reload_task.cancel()
            await asyncio.gather(self._skill_reload_task, return_exceptions=True)
            self._skill_reload_task = None
        if self._session_sweeper_task is not None:
            self._session_sweeper_task.cancel()
            await asyncio.gather(self._session_sweeper_task, return_exceptions=True)
            self._session_sweeper_task = None
        await self._drain_active_sessions(reason="server_shutdown")
        await self._await_background_tasks(self._shutdown_drain_timeout_seconds)

    async def _await_session_close(self, session_id: str) -> None:
        """Block until any in-flight close of *session_id* has finished."""
        deadline = time.monotonic() + _SHUTDOWN_DRAIN_TIMEOUT_SECONDS
        while session_id in self._closing_sessions and time.monotonic() < deadline:
            await asyncio.sleep(0.01)

    async def _close_session(self, session_id: str, reason: str = "explicit") -> None:
        """Flush a session: finalize pending turn feedback, upload session data, clean up state."""
        if not session_id:
            return
        if session_id in self._closing_sessions:
            # Wait for the in-flight close instead of returning immediately:
            # returning let a caller start a new conversation on the same id
            # whose turns were then wiped and uploaded by the first close.
            await self._await_session_close(session_id)
            return
        self._closing_sessions.add(session_id)
        try:
            self._flush_pending_record(session_id, None)
            pending = self._pending_turn_data.get(session_id, {})
            prm_tasks = self._prm_tasks.setdefault(session_id, {})
            if self.config.use_prm and self.prm_scorer:
                for turn_num, turn_data in list(pending.items()):
                    if turn_num in prm_tasks:
                        continue
                    prm_task = asyncio.create_task(
                        self.prm_scorer.evaluate(
                            turn_data.get("response_text", ""),
                            turn_data.get("prompt_text", ""),
                            session_id=session_id,
                            turn_num=turn_num,
                        )
                    )
                    prm_task.add_done_callback(self._task_done_cb)
                    prm_task.add_done_callback(
                        lambda _t, sid=session_id, tnum=turn_num: self._on_prm_done_record_only(sid, tnum, _t)
                    )
                    prm_tasks[turn_num] = prm_task
            active_prm_tasks = list(prm_tasks.values())
            if active_prm_tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*active_prm_tasks, return_exceptions=True),
                        timeout=_SHUTDOWN_DRAIN_TIMEOUT_SECONDS,
                    )
                except asyncio.TimeoutError:
                    logger.warning("[SessionDetect] PRM drain timed out for session=%s", session_id)
            for turn_num in sorted(list(pending.keys())):
                turn_data = pending.pop(turn_num)
                prm_result = turn_data.pop("prm_result", None)
                prm_task = prm_tasks.get(turn_num)
                if prm_result is None and prm_task is not None and prm_task.done():
                    try:
                        prm_result = prm_task.result()
                    except (asyncio.CancelledError, Exception):
                        prm_result = None
                prm_tasks.pop(turn_num, None)
                await self._finalize_turn_feedback(
                    turn_num,
                    turn_data,
                    session_id,
                    prm_result,
                )
            eff = self._session_scored_turns.pop(session_id, 0)
            self._turn_counts.pop(session_id, None)
            self._user_turn_counts.pop(session_id, None)
            self._pending_turn_data.pop(session_id, None)
            prm_tasks = self._prm_tasks.pop(session_id, {})
            for task in prm_tasks.values():
                if isinstance(task, asyncio.Task) and not task.done():
                    task.cancel()
            logger.info(
                "[SessionDetect] closed session=%s reason=%s (scored_turns=%d)",
                session_id,
                reason,
                eff,
            )
            if self.skill_manager:
                self.skill_manager._save_stats()
            turns = self._session_turns.pop(session_id, [])
            modified_skill_names = _extract_modified_skill_names(turns)
            if turns and self.config.sharing_enabled:
                upload = asyncio.create_task(self._upload_session_data(session_id, turns))
                upload.add_done_callback(self._task_done_cb)
                if reason == "server_shutdown":
                    # A restart faster than the upload used to lose whole
                    # sessions: the drain cancelled the task and the drained
                    # state was gone, so there was no retry. Let the last
                    # session data reach the server before we exit.
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(upload),
                            timeout=self._shutdown_drain_timeout_seconds,
                        )
                    except (asyncio.TimeoutError, Exception) as e:
                        logger.warning("[SessionDetect] final session upload incomplete for %s: %s", session_id, e)
                else:
                    self._background_tasks.add(upload)
            if self.config.sharing_enabled:
                self._safe_create_task(self._pull_skills_from_cloud(skip_names=modified_skill_names))
            self._release_session_responses(session_id)
            self._session_last_active.pop(session_id, None)
            for key, meta in list(self._tui_session_meta.items()):
                if isinstance(meta, dict) and meta.get("session_id") == session_id:
                    self._tui_session_meta.pop(key, None)
        finally:
            self._closing_sessions.discard(session_id)

    # ------------------------------------------------------------------ #
    # Record helpers                                                       #
    # ------------------------------------------------------------------ #

    def _flush_pending_record(self, session_id: str, next_state):
        """Write out the buffered record for *session_id* and fire PRM."""
        rec = self._pending_records.pop(session_id, None)
        if rec is None:
            return
        rec["next_state"] = next_state
        if next_state:
            ns_role = next_state.get("role", "?")
            ns_content = _flatten_message_content(next_state.get("content"))
            logger.info(
                f"{_GREEN}[OpenClaw] session={session_id} turn={rec['turn']} "
                f"next_state role={ns_role} len={len(ns_content)}: "
                f"{ns_content[:200]}{_RESET}"
            )
            self._fire_prm_scoring(
                session_id,
                rec["turn"],
                rec["response_text"],
                rec.get("instruction_text", ""),
                next_state,
            )
        if self._record_file:
            try:
                with open(self._record_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except OSError as e:
                logger.warning("[OpenClaw] failed to write record: %s", e)

    def _buffer_record(
        self, session_id: str, turn_num: int, messages: list, prompt_text: str, response_text: str, tool_calls: list
    ):
        if not self._record_file:
            return
        instruction_text = _extract_last_user_instruction(messages)
        self._pending_records[session_id] = {
            "session_id": session_id,
            "turn": turn_num,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "messages": messages,
            "instruction_text": instruction_text,
            "prompt_text": prompt_text,
            "response_text": response_text,
            "tool_calls": tool_calls or None,
        }

    def _append_prm_record(self, session_id: str, turn_num: int, score: float, votes: list):
        if not self._prm_record_file:
            return
        try:
            with open(self._prm_record_file, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "session_id": session_id,
                            "turn": turn_num,
                            "score": score,
                            "votes": votes,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        except OSError as e:
            logger.warning("[OpenClaw] failed to write PRM record: %s", e)

    def purge_record_files(self):
        """Clear all record JSONL files."""
        for path, label in [
            (self._record_file, "record"),
            (self._prm_record_file, "PRM record"),
        ]:
            if not path:
                continue
            try:
                open(path, "w").close()
                logger.info("[OpenClaw] %s file purged: %s", label, path)
            except OSError as e:
                logger.warning("[OpenClaw] failed to purge %s file: %s", label, e)

    # ------------------------------------------------------------------ #
    # PRM scoring                                                          #
    # ------------------------------------------------------------------ #

    def _fire_prm_scoring(
        self,
        session_id: str,
        turn_num: int,
        response_text: str,
        instruction_text: str,
        next_state,
        finalize_ready_turns: bool = True,
    ):
        if not self.prm_scorer or not next_state:
            return
        inst_text = instruction_text or ""
        task = asyncio.create_task(
            self.prm_scorer.evaluate(response_text, inst_text, session_id=session_id, turn_num=turn_num)
        )
        task.add_done_callback(self._task_done_cb)
        if finalize_ready_turns:
            task.add_done_callback(lambda _t: self._on_prm_done(session_id, turn_num, _t))
        else:
            task.add_done_callback(lambda _t: self._on_prm_done_record_only(session_id, turn_num, _t))
        self._prm_tasks.setdefault(session_id, {})[turn_num] = task
        td = self._pending_turn_data.get(session_id, {}).get(turn_num)
        if td is not None:
            td["has_next_state"] = True

    def _apply_prm_result(
        self,
        session_id: str,
        turn_num: int,
        prm_result: Optional[dict],
    ) -> None:
        score = prm_result.get("score", 0.0) if prm_result else 0.0
        turns = self._session_turns.get(session_id, [])
        # turn_num is 1-based; list index is 0-based
        idx = turn_num - 1
        if 0 <= idx < len(turns):
            turns[idx]["prm_score"] = score
            injected = turns[idx].get("injected_skills", [])
            if injected and self.skill_manager:
                self.skill_manager.record_feedback(injected, score)
            read = turns[idx].get("read_skills", [])
            if read and self.skill_manager:
                read_names = [r["skill_name"] for r in read if isinstance(r, dict) and r.get("skill_name")]
                if read_names:
                    self.skill_manager.record_feedback(read_names, score)
        pending_turn = self._pending_turn_data.get(session_id, {}).get(turn_num)
        if isinstance(pending_turn, dict):
            pending_turn["prm_result"] = prm_result

    def _on_prm_done(self, session_id: str, turn_num: int, task: asyncio.Task):
        """Callback after PRM scoring completes — write score back and update skill stats."""
        if task.cancelled():
            return
        try:
            prm_result = task.result()
        except Exception:
            return
        self._apply_prm_result(session_id, turn_num, prm_result)
        if session_id in self._closing_sessions:
            return
        self._maybe_finalize_ready_turns(session_id)

    def _on_prm_done_record_only(self, session_id: str, turn_num: int, task: asyncio.Task):
        """Callback used for close-session PRM tasks; records score only."""
        if task.cancelled():
            return
        try:
            prm_result = task.result()
        except Exception:
            return
        self._apply_prm_result(session_id, turn_num, prm_result)

    # ------------------------------------------------------------------ #
    # Request handling                                                     #
    # ------------------------------------------------------------------ #

    def _read_cached_system_prompt(self, source_fingerprint: str = "") -> str:
        """Return a cached compressed prompt for the *same* source prompt.

        The cache used to be one process-wide file per claw/provider pair, so
        whichever session arrived first had its compressed instructions written
        once and then substituted into every later request from every session —
        including after a restart. The fingerprint ties the cache to the text it
        was actually derived from.
        """
        try:
            with open(self._system_prompt_cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            cached = data.get("compressed_system_prompt", "")
            if not isinstance(cached, str) or not cached:
                return ""
            cached_source = str(data.get("source_fingerprint") or "")
            if source_fingerprint and cached_source != source_fingerprint:
                logger.info(
                    "[OpenClaw] system prompt cache miss: source prompt differs from cached source"
                )
                return ""
            return cached
        except Exception:
            return ""

    def _write_cached_system_prompt(self, prompt: str, source_fingerprint: str = ""):
        try:
            with open(self._system_prompt_cache_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "compressed_system_prompt": prompt,
                        "source_fingerprint": source_fingerprint,
                    },
                    f,
                    ensure_ascii=False,
                )
        except Exception as e:
            logger.warning("[OpenClaw] failed to write system prompt cache: %s", e)

    async def _handle_request(
        self,
        body: dict[str, Any],
        session_id: str,
        turn_type: str,
        session_done: bool,
    ) -> dict[str, Any]:
        protocol = str(body.get("_skillclaw_protocol") or "").strip()
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=400, detail="messages must be a non-empty list")
        # Keep the session out of the idle sweeper's reach for as long as this
        # request runs, and re-arm the clock when it finishes.
        with self._in_flight_request(session_id):
            return await self._handle_request_inner(
                body,
                session_id=session_id,
                turn_type=turn_type,
                session_done=session_done,
                protocol=protocol,
                messages=messages,
            )

    async def _handle_request_inner(
        self,
        body: dict[str, Any],
        *,
        session_id: str,
        turn_type: str,
        session_done: bool,
        protocol: str,
        messages: list,
    ) -> dict[str, Any]:
        rewritten = 0
        for msg in messages:
            if (
                isinstance(msg, dict)
                and msg.get("role") == "user"
                and isinstance(msg.get("content"), str)
                and msg.get("content", "").startswith("A new chat session just started.")
            ):
                rewritten += 1
        if rewritten:
            logger.info("[OpenClaw] rewrote %d /new bootstrap user prompt(s) for provider safety", rewritten)

        def _prompt_len(msgs):
            return _estimate_openai_body_input_tokens({"messages": msgs, "tools": body.get("tools")})

        # Compress verbose system prompts (OpenClaw only).  Non-OpenClaw
        # agents send short or no system prompts; compressing them wastes an
        # LLM call and the cached OpenClaw prompt can trigger content filters.
        cached_system = ""
        if self._compress_system_prompt:
            raw_system = ""
            for m in messages:
                if isinstance(m, dict) and m.get("role") == "system":
                    raw_system = _flatten_message_content(m.get("content"))
                    break
            # The cache is keyed on the prompt it was derived from, so another
            # session's (or another project's) instructions can never be
            # substituted into this request.
            fingerprint = _system_prompt_fingerprint(raw_system) if raw_system else ""
            cached_system = self._read_cached_system_prompt(fingerprint)
            if not cached_system:
                if raw_system:
                    try:
                        cached_system = await asyncio.to_thread(
                            run_llm,
                            [{"role": "user", "content": raw_system}],
                            self.config,
                        )
                        cached_system = (cached_system or raw_system).strip()
                    except Exception as e:
                        logger.warning(
                            "[OpenClaw] system prompt compression failed: %s — using raw system prompt",
                            e,
                        )
                        cached_system = raw_system.strip()
                    self._write_cached_system_prompt(cached_system, fingerprint)

            if cached_system:
                for m in messages:
                    if isinstance(m, dict) and m.get("role") == "system":
                        m["content"] = cached_system

        restored_reasoning = _restore_missing_reasoning_content(
            messages,
            self._session_turns.get(session_id, []),
        )
        if restored_reasoning:
            logger.info(
                "[OpenClaw] restored reasoning_content on %d prior assistant tool-call message(s)",
                restored_reasoning,
            )

        tools = body.get("tools")

        # Inject skills into system message for main turns
        injected_skills: list[str] = []
        if self.skill_manager and turn_type == "main":
            messages, injected_skills = self._inject_skills(messages)
        if self._compress_system_prompt and cached_system:
            logger.info(
                "[OpenClaw] system prompt cached len=%d",
                _prompt_len([{"role": "system", "content": cached_system}]),
            )

        forward_body = {k: v for k, v in body.items() if k not in _NON_STANDARD_BODY_KEYS}
        # Cap BEFORE sizing the prompt budget. Deriving the budget from the raw
        # client value let a large max_tokens make max_prompt negative, which
        # skipped truncation entirely and forwarded an over-limit prompt — the
        # exact failure _cap_completion_token_fields exists to prevent.
        _cap_completion_token_fields(forward_body)
        requested_completion = _coerce_int(
            forward_body.get("max_completion_tokens", forward_body.get("max_tokens")),
            2048,
        )

        # Truncate to fit within max_context_tokens (keep system + most-recent
        # messages). Reserve room for the completion, but never let a large
        # client max_tokens drive the prompt budget to zero: that silently
        # disabled truncation and forwarded an over-limit prompt upstream.
        reserved = min(requested_completion, max(0, self.config.max_context_tokens - _MIN_PROMPT_TOKENS))
        max_prompt = self.config.max_context_tokens - reserved
        if max_prompt > 0:
            messages = self._truncate_messages(messages, tools, max_prompt)

        forward_body["stream"] = False
        forward_body.pop("stream_options", None)
        if "model" not in forward_body:
            forward_body["model"] = self._served_model
        # Single choke point: sanitize after skill injection and truncation so
        # both a client's malformed body and a truncation cut are made valid.
        forward_body["messages"] = _sanitize_forward_messages(messages)

        output = await self._forward_to_llm(forward_body)
        output["model"] = forward_body.get("model") or self._served_model

        # .get(k, default) only fires when the key is absent; a present-but-empty
        # list (content filter, truncated upstream response) would IndexError.
        choices = output.get("choices") or [{}]
        choice = choices[0] if choices else {}
        assistant_msg = choice.get("message", {})
        if not isinstance(assistant_msg, dict):
            assistant_msg = {"role": "assistant", "content": _flatten_message_content(assistant_msg)}

        raw_tool_calls = assistant_msg.get("tool_calls") or []
        tool_calls = list(raw_tool_calls) if isinstance(raw_tool_calls, list) else []

        raw_content = assistant_msg.get("content")
        if isinstance(raw_content, list):
            part_text, part_tool_calls = _normalize_assistant_content_parts(raw_content)
            content = part_text
            tool_calls.extend(part_tool_calls)
        else:
            content = _flatten_message_content(raw_content)

        # Upstream models sometimes emit tool calls as text tags instead of
        # structured `message.tool_calls`; parse and normalize both sources.
        clean_content, text_tool_calls = _extract_tool_calls_from_text(content)
        if text_tool_calls:
            content = clean_content
            tool_calls.extend(text_tool_calls)
        tool_calls = _deduplicate_tool_calls(tool_calls)

        assistant_msg["content"] = content
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        else:
            assistant_msg.pop("tool_calls", None)
        choice["message"] = assistant_msg
        if isinstance(output.get("choices"), list) and output["choices"]:
            output["choices"][0] = choice
        else:
            output["choices"] = [choice]

        reasoning = assistant_msg.get("reasoning_content") or ""

        logger.info(f"{_YELLOW}[OpenClaw] [{turn_type}] session={session_id} prompt_msgs={len(messages)}{_RESET}")
        logger.info(
            f"{_RED}[OpenClaw] [{turn_type}] session={session_id} "
            f"thinking={len(reasoning)} chars, response:\n{content}{_RESET}"
        )
        if tool_calls:
            logger.info("[OpenClaw] tool_calls: %s", json.dumps(tool_calls, ensure_ascii=False)[:500])

        if turn_type == "main":
            tool_results = _extract_recent_tool_results(messages)
            prev_turns = self._session_turns.get(session_id, [])
            if tool_results and prev_turns:
                raw_tool_results = _extract_recent_tool_result_messages(messages)
                _merge_tool_error_info(prev_turns[-1], tool_results, raw_tool_results)

            if session_id in self._pending_records and messages:
                self._flush_pending_record(session_id, messages[-1])

            response_msg = dict(assistant_msg)
            if response_msg.get("content") is None:
                response_msg["content"] = ""

            skill_path_map = self.skill_manager.get_skill_path_map() if self.skill_manager else {}
            read_skills = _extract_read_skills_from_tool_calls(
                tool_calls,
                skill_path_map,
            )
            modified_skills = _extract_modified_skills_from_tool_calls(
                tool_calls,
                skill_path_map,
            )
            tool_summaries = _build_tool_summaries(tool_calls)
            if read_skills:
                logger.info(
                    "[SkillManager] model read %d skill(s): %s",
                    len(read_skills),
                    ", ".join(r.get("skill_name", "?") for r in read_skills),
                )
            if modified_skills:
                logger.info(
                    "[SkillManager] model modified %d skill(s): %s",
                    len(modified_skills),
                    ", ".join(r.get("skill_name", "?") for r in modified_skills),
                )

            user_instruction = _extract_last_user_instruction(messages)
            self._turn_counts[session_id] = self._turn_counts.get(session_id, 0) + 1
            turn_num = self._turn_counts[session_id]
            prompt_text = "\n".join(
                f"{m.get('role', '?')}: {_flatten_message_content(m.get('content', ''))}"
                for m in messages
                if isinstance(m, dict)
            )
            response_text = content or (json.dumps(tool_calls, ensure_ascii=False) if tool_calls else "")
            self._buffer_record(session_id, turn_num, messages, prompt_text, response_text, tool_calls)
            raw_turn_kind = _classify_raw_turn_kind(protocol, content, tool_calls)
            turn_record = {
                "turn_num": turn_num,
                "raw_turn_kind": raw_turn_kind,
                "prompt_text": user_instruction,
                "response_text": response_text,
                "reasoning_content": reasoning or None,
                "tool_calls": tool_calls,
                "read_skills": read_skills,
                "modified_skills": modified_skills,
                "tool_results": tool_summaries,
                "tool_results_raw": [],
                "tool_observations": [],
                "tool_errors": [],
                "injected_skills": injected_skills,
                "prm_score": None,
            }
            self._session_turns.setdefault(session_id, []).append(turn_record)
            if _is_user_turn_boundary(raw_turn_kind):
                user_turn_num = self._next_user_turn_num(session_id)
                turn_record["user_turn_num"] = user_turn_num
                self._maybe_upload_session_snapshot(session_id, user_turn_num)
            self._pending_turn_data.setdefault(session_id, {})[turn_num] = {
                "prompt_text": prompt_text,
                "response_text": response_text,
            }
            logger.info(
                "[OpenClaw] MAIN session=%s turn=%d user_turn=%s kind=%s prompt_est_tokens=%d response_chars=%d",
                session_id,
                turn_num,
                turn_record.get("user_turn_num", "-"),
                raw_turn_kind,
                _estimate_openai_body_input_tokens({"messages": messages, "tools": tools}),
                len(response_text),
            )
            self._maybe_finalize_ready_turns(session_id)
        else:
            logger.info("[OpenClaw] SIDE session=%s -> skipped (side-channel turn)", session_id)

        if session_done:
            await self._close_session(session_id)

        output["session_id"] = session_id
        # Report what was actually forwarded so the Responses store can record
        # the trimmed history rather than re-inflating the full one.
        return {"response": output, "forwarded_messages": forward_body.get("messages")}

    # ------------------------------------------------------------------ #
    # LLM forwarding                                                       #
    # ------------------------------------------------------------------ #

    async def _forward_to_llm(self, body: dict[str, Any]) -> dict[str, Any]:
        """Forward to a real LLM API.

        Supports providers:
          - ``"openai"`` (default) — any OpenAI-compatible ``/v1/chat/completions`` endpoint.
          - ``"openrouter"`` — OpenRouter gateway (OpenAI-compatible + routing extensions).
          - ``"bedrock"`` — AWS Bedrock Converse API via :class:`BedrockChatClient`.
        """
        if self.config.llm_provider == "bedrock":
            return await self._forward_to_llm_bedrock(body)
        return await self._forward_to_llm_openai(body)

    def _responses_native_enabled(self) -> bool:
        """Return whether /v1/responses should be forwarded as Responses API."""
        return str(getattr(self.config, "llm_api_mode", "chat") or "chat").lower() == "responses"

    def _codex_oauth_enabled(self) -> bool:
        """Return whether upstream auth should use ChatGPT-account OAuth."""
        return str(getattr(self.config, "llm_provider", "") or "").strip().lower() == "codex_oauth"

    async def _build_upstream_auth_headers(self, api_base: str) -> dict[str, str]:
        """Build upstream auth headers for the configured provider.

        For ``codex_oauth`` this resolves a live ChatGPT-account OAuth token
        (refreshing it only when actually expired) plus the harness-identity
        headers OpenAI's Codex endpoint requires.  Every other provider keeps
        the historical static-API-key behavior.

        Async because the OAuth path does blocking file I/O and, on an expired
        token, a blocking HTTP refresh with a 20s timeout. Running that inline
        on the event loop stalls every concurrent request behind it.
        """
        if not self._codex_oauth_enabled():
            if self.config.llm_api_key:
                return {"Authorization": f"Bearer {self.config.llm_api_key}"}
            return {}

        from . import codex_oauth

        try:
            return await asyncio.to_thread(codex_oauth.build_auth_headers, api_base)
        except codex_oauth.CodexAuthError as e:
            detail = f"Codex OAuth auth failed: {e}"
            if getattr(e, "relogin_required", False):
                detail += " — run `hermes auth` (or `codex`) to re-authenticate."
            logger.error("[CodexOAuth] %s", detail)
            raise HTTPException(status_code=401, detail=detail) from e

    async def _prepare_responses_forward(
        self,
        body: dict[str, Any],
        *,
        stream: bool,
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        """Build URL, body, and headers for native Responses forwarding.

        Native mode intentionally keeps Responses-only tools (custom, web_search,
        namespace, etc.) untouched instead of converting the request to chat.
        """
        api_base = self.config.llm_api_base.rstrip("/")
        if not api_base:
            raise HTTPException(
                status_code=503,
                detail="llm_api_base is not configured. Run 'skillclaw setup' first.",
            )

        send_body = {k: v for k, v in body.items() if k not in _NON_STANDARD_BODY_KEYS}
        _cap_completion_token_fields(send_body)
        send_body["model"] = self.config.llm_model_id or body.get("model", "")
        send_body["stream"] = stream

        # Native Responses had no truncation, so an over-budget conversation was
        # forwarded verbatim and rejected upstream. Use the same reserve policy
        # as the chat path.
        raw_input = send_body.get("input")
        if isinstance(raw_input, list) and raw_input:
            requested_completion = _coerce_int(
                send_body.get("max_output_tokens", send_body.get("max_completion_tokens")),
                2048,
            )
            reserved = min(requested_completion, max(0, self.config.max_context_tokens - _MIN_PROMPT_TOKENS))
            max_prompt = self.config.max_context_tokens - reserved
            if max_prompt > 0:
                send_body["input"] = _truncate_responses_input(raw_input, max_prompt)

        headers = await self._build_upstream_auth_headers(api_base)
        return f"{api_base}/responses", send_body, headers

    def _prepare_native_responses_body(self, body: dict[str, Any], *, turn_type: str) -> dict[str, Any]:
        """Apply non-destructive SkillClaw hooks before native Responses forwarding."""
        prepared = dict(body)
        self._prepare_native_responses_body_inplace(prepared, turn_type=turn_type)
        return prepared

    def _prepare_native_responses_body_inplace(self, body: dict[str, Any], *, turn_type: str) -> list[str]:
        """Inject skills into a Responses body in-place. Returns injected skill names."""
        if not self.skill_manager or turn_type != "main":
            return []

        try:
            self.skill_manager.refresh_if_changed()
        except Exception as e:
            logger.warning("[SkillManager] failed to refresh local skills: %s", e)

        skill_text = self.skill_manager.build_injection_prompt(
            max_chars=getattr(self.config, "max_skills_prompt_chars", 30_000),
        )
        if not skill_text:
            return []

        all_skills = self.skill_manager.get_all_skills()
        skill_names = [s.get("name", "unknown_skill") for s in all_skills if isinstance(s, dict)]
        logger.info(
            "[SkillManager] listing %d skills in Codex Responses instructions: %s",
            len(skill_names),
            ", ".join(skill_names)[:400],
        )
        self.skill_manager.record_injection(skill_names)

        existing = _normalize_responses_content(body.get("instructions", ""))
        body["instructions"] = (existing + "\n\n" + skill_text).strip() if existing else skill_text
        return skill_names

    def _record_responses_turn(
        self,
        session_id: str,
        request_body: dict[str, Any],
        response_payload: dict[str, Any],
        *,
        turn_type: str,
        injected_skills: list[str],
        session_done: bool,
    ) -> None:
        """Record a Responses API turn into the session tracking system."""
        if not session_id:
            return
        self._touch_session(session_id)
        prompt_text = _normalize_responses_content(request_body.get("instructions", ""))
        inp = request_body.get("input")
        if isinstance(inp, str):
            prompt_text = (prompt_text + "\n" + inp).strip() if prompt_text else inp
        elif isinstance(inp, list):
            user_parts = []
            for item in inp:
                if isinstance(item, dict) and item.get("role") == "user":
                    user_parts.append(_normalize_responses_content(item.get("content", "")))
            if user_parts:
                joined = " ".join(user_parts)
                prompt_text = (prompt_text + "\n" + joined).strip() if prompt_text else joined
        response_parts = []
        for item in response_payload.get("output", []):
            if not isinstance(item, dict):
                continue
            if item.get("type") == "message":
                for part in item.get("content", []):
                    if isinstance(part, dict) and part.get("type") == "output_text":
                        response_parts.append(part.get("text", ""))
            elif item.get("type") == "function_call":
                name = item.get("name", "")
                args = str(item.get("arguments", ""))[:500]
                response_parts.append(f"[tool:{name}] {args}")
        response_text = "\n".join(response_parts)
        turns = self._session_turns.setdefault(session_id, [])
        turn_num = len(turns) + 1
        turn_record = {
            "turn_num": turn_num,
            "raw_turn_kind": "final" if turn_type == "main" else "side",
            "prompt_text": prompt_text[:2000],
            "response_text": response_text[:2000],
            "injected_skills": injected_skills,
            "prm_score": None,
        }
        turns.append(turn_record)
        if turn_type == "main":
            user_turn_num = self._next_user_turn_num(session_id)
            turn_record["user_turn_num"] = user_turn_num
            self._maybe_upload_session_snapshot(session_id, user_turn_num)
        logger.info(
            "[Codex] %s session=%s turn=%d user_turn=%s prompt=%d chars response=%d chars skills=%s",
            turn_type,
            session_id,
            turn_num,
            turn_record.get("user_turn_num", "-"),
            len(prompt_text),
            len(response_text),
            ",".join(injected_skills) if injected_skills else "(none)",
        )
        if session_done:
            self._safe_create_task(self._close_session(session_id, reason="codex_session_done"))

    async def _forward_to_llm_responses(self, body: dict[str, Any]) -> dict[str, Any]:
        """Forward a Codex Responses payload to an upstream Responses API."""
        import httpx

        url, send_body, headers = await self._prepare_responses_forward(body, stream=False)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=_llm_request_timeout_seconds()) as client:
                    resp = await client.post(
                        url,
                        json=send_body,
                        headers=headers,
                    )
                    resp.raise_for_status()
                    return resp.json()
            except httpx.HTTPStatusError as e:
                response_text = e.response.text[:200]
                retryable = _is_retryable_status(e.response.status_code)
                if not retryable or attempt >= max_retries - 1:
                    logger.error("[OpenClaw] upstream Responses error: %s %s", e.response.status_code, response_text)
                    if not retryable:
                        raise HTTPException(
                            status_code=e.response.status_code,
                            detail=f"Upstream Responses rejected request: {response_text}",
                        ) from e
                    raise HTTPException(status_code=502, detail=f"Upstream Responses error: {e}") from e
                wait = min(2**attempt + random.uniform(0, 1), 10)
                if e.response.status_code == 429:
                    wait = _retry_after_seconds(e.response, wait)
                logger.warning(
                    "[OpenClaw] upstream Responses error (attempt %d/%d), retrying in %.1fs: %s %s",
                    attempt + 1,
                    max_retries,
                    wait,
                    e.response.status_code,
                    response_text,
                )
                await asyncio.sleep(wait)
                continue
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = min(2**attempt + random.uniform(0, 1), 10)
                    logger.warning(
                        "[OpenClaw] Responses forward failed (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1,
                        max_retries,
                        wait,
                        e,
                    )
                    await asyncio.sleep(wait)
                    continue
                logger.error("[OpenClaw] Responses forward failed: %s", e, exc_info=True)
                raise HTTPException(status_code=502, detail=f"Responses forward error: {e}") from e

    async def _stream_and_track_responses(
        self,
        body: dict[str, Any],
        *,
        record_body: dict[str, Any] | None = None,
        session_id: str,
        turn_type: str,
        injected_skills: list[str],
        session_done: bool,
    ):
        """Wrap _stream_llm_responses: passthrough SSE + parse response.completed inline."""
        tracked = False
        buf = ""
        # Incremental UTF-8 decoder: holds a codepoint split across two chunks.
        decoder = codecs.getincrementaldecoder("utf-8")()
        flushed = False
        output_items: dict[int, dict[str, Any]] = {}
        output_text_parts: dict[tuple[int, int], str] = {}

        def ensure_message_item(output_index: int) -> dict[str, Any]:
            item = output_items.setdefault(
                output_index,
                {
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [],
                },
            )
            content = item.setdefault("content", [])
            if not isinstance(content, list):
                item["content"] = []
            return item

        def apply_output_text(output_index: int, content_index: int, text: str) -> None:
            item = ensure_message_item(output_index)
            content = item.setdefault("content", [])
            while len(content) <= content_index:
                content.append({"type": "output_text", "text": "", "annotations": []})
            part = content[content_index]
            if isinstance(part, dict):
                part["type"] = part.get("type") or "output_text"
                part["text"] = text
                part.setdefault("annotations", [])

        def parse_responses_stream_event(data: dict[str, Any]) -> dict[str, Any] | None:
            event_type = data.get("type")
            output_index = int(data.get("output_index", 0) or 0)
            content_index = int(data.get("content_index", 0) or 0)

            if event_type == "response.output_item.added":
                item = data.get("item")
                if isinstance(item, dict):
                    output_items[output_index] = item
            elif event_type == "response.output_item.done":
                item = data.get("item")
                if isinstance(item, dict):
                    output_items[output_index] = item
            elif event_type == "response.output_text.delta":
                key = (output_index, content_index)
                output_text_parts[key] = output_text_parts.get(key, "") + str(data.get("delta") or "")
                apply_output_text(output_index, content_index, output_text_parts[key])
            elif event_type == "response.output_text.done":
                text = str(data.get("text") or output_text_parts.get((output_index, content_index), ""))
                output_text_parts[(output_index, content_index)] = text
                apply_output_text(output_index, content_index, text)
            elif event_type == "response.content_part.done":
                part = data.get("part")
                if isinstance(part, dict) and part.get("type") == "output_text":
                    text = str(part.get("text") or output_text_parts.get((output_index, content_index), ""))
                    output_text_parts[(output_index, content_index)] = text
                    apply_output_text(output_index, content_index, text)
            elif event_type == "response.completed":
                response_payload = data.get("response") if isinstance(data.get("response"), dict) else dict(data)
                if output_items and not response_payload.get("output"):
                    response_payload = {
                        **response_payload,
                        "output": [item for _, item in sorted(output_items.items())],
                    }
                return response_payload
            return None

        try:
            async for chunk in self._stream_llm_responses(body):
                if not tracked:
                    try:
                        # Decode a copy for tracking only: the yielded value must
                        # stay exactly the upstream bytes.
                        if isinstance(chunk, bytes):
                            # A codepoint can straddle two upstream chunks;
                            # decoding each chunk with errors="ignore" dropped
                            # the partial bytes and corrupted the recorded turn.
                            text = decoder.decode(chunk)
                        elif not flushed:
                            text = decoder.decode(b"", final=True)
                            flushed = True
                        else:
                            text = chunk if isinstance(chunk, str) else ""
                        buf += text
                        while "\n" in buf:
                            line, buf = buf.split("\n", 1)
                            stripped = line.strip()
                            if not stripped.startswith("data: ") and not stripped.startswith("data:"):
                                continue
                            raw = stripped.split(":", 1)[1].strip()
                            if raw == "[DONE]":
                                continue
                            try:
                                data = json.loads(raw)
                            except Exception:
                                continue
                            response_payload = parse_responses_stream_event(data) if isinstance(data, dict) else None
                            if response_payload is not None:
                                self._record_responses_turn(
                                    session_id,
                                    record_body or body,
                                    response_payload,
                                    turn_type=turn_type,
                                    injected_skills=injected_skills,
                                    session_done=session_done,
                                )
                                tracked = True
                                break
                    except Exception:
                        pass
                yield chunk
        except Exception as e:
            # Headers are already on the wire, so a failure here cannot become an
            # HTTP status. Emit a terminal error event and close the stream
            # cleanly so the client can tell truncation from success instead of
            # seeing a dead socket with no terminator.
            logger.error("[OpenClaw] Responses stream ended early: %s", e)
            yield _sse_chunk(_stream_error_event(e, model=str(body.get("model") or "")))
            yield "data: [DONE]\n\n"

    async def _stream_llm_responses(self, body: dict[str, Any]):
        """Passthrough upstream Responses SSE without aggregating or rewriting events."""
        import httpx

        url, send_body, headers = await self._prepare_responses_forward(body, stream=True)
        try:
            async with httpx.AsyncClient(timeout=_llm_request_timeout_seconds()) as client:
                async with client.stream("POST", url, json=send_body, headers=headers) as resp:
                    resp.raise_for_status()
                    async for chunk in resp.aiter_bytes():
                        if chunk:
                            yield chunk
        except httpx.HTTPStatusError as e:
            response_text = e.response.text[:200]
            logger.error("[OpenClaw] upstream Responses stream error: %s %s", e.response.status_code, response_text)
            raise HTTPException(status_code=502, detail=f"Upstream Responses stream error: {e}") from e
        except Exception as e:
            logger.error("[OpenClaw] Responses stream failed: %s", e, exc_info=True)
            raise HTTPException(status_code=502, detail=f"Responses stream error: {e}") from e

    def _chat_body_to_responses(self, body: dict[str, Any]) -> dict[str, Any]:
        """Translate an OpenAI chat-completions body into a Responses body.

        Used when a chat-only client (e.g. the evolve server's ``AsyncLLMClient``)
        talks to a proxy whose upstream exposes ONLY ``/responses`` -- notably the
        Codex backend under ``codex_oauth``.  System/developer messages become
        ``instructions``; the remaining turns become typed ``input`` items.
        """
        instructions: list[str] = []
        input_items: list[dict[str, Any]] = []

        for message in body.get("messages") or []:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "user")
            content = message.get("content")
            if isinstance(content, list):
                text = "".join(
                    str(part.get("text") or "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") in {"text", "input_text", "output_text"}
                )
            else:
                text = str(content or "")
            if role == "tool":
                # A tool result is a function_call_output item, not a message.
                # Emitting it as a message made the upstream see a bare
                # role:tool turn and the assistant's function_call was never
                # replayed, so the whole tool round was lost.
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": str(message.get("tool_call_id") or ""),
                        "output": text or str(content or ""),
                    }
                )
                continue
            if role == "assistant" and message.get("tool_calls"):
                # Text first (may be empty), then one function_call per tool.
                if text:
                    input_items.append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": text}],
                        }
                    )
                for call in message["tool_calls"]:
                    if not isinstance(call, dict):
                        continue
                    raw_function = call.get("function")
                    function = raw_function if isinstance(raw_function, dict) else {}
                    input_items.append(
                        {
                            "type": "function_call",
                            "call_id": str(call.get("id") or ""),
                            "name": str(function.get("name") or ""),
                            "arguments": str(function.get("arguments") or ""),
                        }
                    )
                continue
            if not text:
                continue
            if role in {"system", "developer"}:
                instructions.append(text)
                continue
            # Responses uses input_text for user turns and output_text for
            # assistant turns; mixing them up is a 400 from the backend.
            part_type = "output_text" if role == "assistant" else "input_text"
            input_items.append({"type": "message", "role": role, "content": [{"type": part_type, "text": text}]})

        send_body: dict[str, Any] = {
            "model": self.config.llm_model_id or body.get("model", ""),
            "input": input_items,
            "stream": False,
            "store": False,
        }
        if instructions:
            send_body["instructions"] = "\n\n".join(instructions)
        # Tools used to be dropped here, so the upstream never learned the
        # client could call anything and returned text only.
        tools = body.get("tools")
        if isinstance(tools, list) and tools:
            send_body["tools"] = _chat_tools_to_responses(tools)
        # NOTE: deliberately no max_output_tokens / temperature passthrough.
        # The Codex backend rejects both ("Unsupported parameter"), and chat
        # clients routinely set them, so silently dropping is the only way the
        # bridge stays usable.
        return send_body

    @staticmethod
    def _responses_payload_to_chat(payload: dict[str, Any], model: str) -> dict[str, Any]:
        """Render a Responses payload in chat-completions shape for chat clients."""
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for item in payload.get("output") or []:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "message":
                for part in item.get("content") or []:
                    if isinstance(part, dict) and part.get("type") == "output_text":
                        text_parts.append(str(part.get("text") or ""))
            elif item.get("type") == "function_call":
                # Only reading `message` items here discarded every tool call,
                # so a tool-using chat client got an empty assistant message.
                call_id = str(item.get("call_id") or item.get("id") or "")
                tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": str(item.get("name") or ""),
                            "arguments": str(item.get("arguments") or ""),
                        },
                    }
                )
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        message: dict[str, Any] = {"role": "assistant", "content": "".join(text_parts)}
        if tool_calls:
            message["tool_calls"] = tool_calls
        return {
            "id": payload.get("id", "chatcmpl-skillclaw"),
            "object": "chat.completion",
            "created": payload.get("created_at", 0),
            "model": payload.get("model", model),
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    # Report upstream truncation honestly; a hardcoded "stop"
                    # made a length-capped answer look like a clean finish.
                    "finish_reason": "tool_calls" if tool_calls else _responses_finish_reason(payload),
                }
            ],
            "usage": {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        }

    async def _forward_chat_via_responses(self, body: dict[str, Any]) -> dict[str, Any]:
        """Serve a chat-completions request from a Responses-only upstream.

        The Codex backend rejects non-streaming Responses calls outright
        ("Stream must be set to true"), so the bridge always streams and
        aggregates the SSE events back into a single chat-shaped payload.
        """
        import httpx

        api_base = self.config.llm_api_base.rstrip("/")
        send_body = self._chat_body_to_responses(body)
        send_body["stream"] = True
        headers = await self._build_upstream_auth_headers(api_base)

        final_payload: dict[str, Any] = {}
        text_parts: list[str] = []
        try:
            async with httpx.AsyncClient(timeout=_llm_request_timeout_seconds()) as client:
                async with client.stream("POST", f"{api_base}/responses", json=send_body, headers=headers) as resp:
                    if resp.status_code >= 400:
                        # Must be read inside the stream context; the body is
                        # otherwise unavailable once the response closes.
                        detail = (await resp.aread()).decode("utf-8", "ignore")[:300]
                        logger.error("[SkillClaw] chat→responses bridge error: %s %s", resp.status_code, detail)
                        raise HTTPException(
                            status_code=502, detail=f"Upstream Responses error {resp.status_code}: {detail}"
                        )
                    async for line in resp.aiter_lines():
                        stripped = line.strip()
                        if not stripped.startswith("data:") :
                            continue
                        raw = stripped.split(":", 1)[1].strip()
                        if raw == "[DONE]":
                            break
                        try:
                            event = json.loads(raw)
                        except Exception:
                            continue
                        if not isinstance(event, dict):
                            continue
                        if isinstance(event.get("error"), (dict, str)):
                            # A mid-stream error frame is a failed generation;
                            # reporting the partial text as a completion hid
                            # upstream crashes from the client.
                            error = event["error"]
                            message = (
                                str(error.get("message") or json.dumps(error)[:200])
                                if isinstance(error, dict)
                                else str(error)
                            )
                            raise _SseStreamError(message)
                        etype = event.get("type")
                        if etype == "response.output_text.delta":
                            text_parts.append(str(event.get("delta") or ""))
                        elif etype == "response.output_item.done":
                            item = event.get("item")
                            # A function_call may arrive only as item events when
                            # response.completed omits the assembled output.
                            if isinstance(item, dict) and item.get("type") == "function_call":
                                _merge_responses_item(final_payload, item)
                        elif etype == "response.completed":
                            candidate = event.get("response")
                            if isinstance(candidate, dict):
                                final_payload = candidate
        except HTTPException:
            raise
        except _SseStreamError as e:
            logger.error("[SkillClaw] chat→responses bridge stream error: %s", e)
            raise HTTPException(status_code=502, detail=f"Upstream Responses stream error: {e.message}") from e
        except Exception as e:
            logger.error("[SkillClaw] chat→responses bridge failed: %s", e, exc_info=True)
            raise HTTPException(status_code=502, detail=f"Responses bridge error: {e}") from e

        chat = self._responses_payload_to_chat(final_payload, send_body.get("model", ""))
        # Prefer the streamed deltas: response.completed occasionally omits the
        # assembled output, which would otherwise yield an empty message.
        if not chat["choices"][0]["message"]["content"] and text_parts:
            chat["choices"][0]["message"]["content"] = "".join(text_parts)
        return chat

    async def _forward_to_llm_openai(self, body: dict[str, Any]) -> dict[str, Any]:
        """Forward to an OpenAI-compatible API."""
        import httpx

        api_base = self.config.llm_api_base.rstrip("/")
        if not api_base:
            raise HTTPException(
                status_code=503,
                detail="llm_api_base is not configured. Run 'skillclaw setup' first.",
            )

        # A Responses-only upstream (the Codex backend) has no
        # /chat/completions route, so chat-shaped callers must be bridged
        # rather than forwarded verbatim into a guaranteed 404.
        if self._responses_native_enabled():
            return await self._forward_chat_via_responses(body)

        # Strip Tinker-specific fields not supported by standard OpenAI APIs
        send_body = {k: v for k, v in body.items() if k not in {"logprobs", "top_logprobs", "stream_options"}}
        send_body["model"] = self.config.llm_model_id or body.get("model", "")
        _normalize_openai_chat_token_fields(send_body)
        send_body["stream"] = False

        headers = await self._build_upstream_auth_headers(api_base)

        # OpenRouter-specific headers and body extensions
        if self.config.llm_provider == "openrouter":
            if self.config.openrouter_app_name:
                headers["X-Title"] = self.config.openrouter_app_name
            if self.config.openrouter_app_url:
                headers["HTTP-Referer"] = self.config.openrouter_app_url
            # Routing strategy
            route = self.config.openrouter_route
            if route and route != "fallback":
                send_body["provider"] = {"sort": route}
            # Fallback model list
            fallback = self.config.openrouter_fallback_models
            if fallback:
                models = [m.strip() for m in fallback.split(",") if m.strip()]
                if models:
                    send_body["models"] = [send_body.get("model", "")] + models
            # Data collection policy
            if self.config.openrouter_data_policy == "deny":
                send_body.setdefault("provider", {})
                send_body["provider"]["data_collection"] = "deny"

        max_retries = 6
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=_llm_request_timeout_seconds()) as client:
                    resp = await client.post(
                        f"{api_base}/chat/completions",
                        json=send_body,
                        headers=headers,
                    )
                    resp.raise_for_status()
                    return resp.json()
            except httpx.HTTPStatusError as e:
                response_text = e.response.text[:200]
                if e.response.status_code == 400 and _is_unsupported_temperature_error(e.response.text):
                    if "temperature" not in send_body or attempt >= max_retries - 1:
                        # Nothing left to drop, or the budget is spent: fall
                        # through to the terminal 502 instead of `continue`,
                        # which would exit the loop and return None.
                        logger.error("[OpenClaw] upstream still rejects temperature after retries")
                        raise HTTPException(
                            status_code=502,
                            detail=f"Upstream LLM rejects temperature: {e}",
                        ) from e
                    logger.info("[OpenClaw] upstream rejects temperature param, retrying without it")
                    send_body.pop("temperature", None)
                    continue
                if e.response.status_code == 400 and "Stream must be set to true" in e.response.text:
                    logger.info("[OpenClaw] upstream requires stream=true, retrying with SSE collection")
                    stream_body = dict(send_body)
                    stream_body["stream"] = True
                    try:
                        async with httpx.AsyncClient(timeout=_llm_request_timeout_seconds()) as client:
                            async with client.stream(
                                "POST",
                                f"{api_base}/chat/completions",
                                json=stream_body,
                                headers=headers,
                            ) as stream_resp:
                                stream_resp.raise_for_status()
                                events = await _collect_sse_chat_events(stream_resp)
                        return _assemble_streaming_chat_completion(
                            events,
                            fallback_model=send_body.get("model", ""),
                        )
                    except _SseStreamError as stream_error:
                        # A mid-stream `error` frame is a failed generation, not
                        # a completion: surface it instead of returning a
                        # truncated answer as a success.
                        logger.error("[OpenClaw] upstream SSE stream reported an error: %s", stream_error)
                        raise HTTPException(
                            status_code=502,
                            detail=f"Upstream LLM stream error: {stream_error.message}",
                        ) from stream_error
                    except httpx.HTTPStatusError as stream_error:
                        logger.error(
                            "[OpenClaw] upstream SSE retry error: %s %s",
                            stream_error.response.status_code,
                            stream_error.response.text[:200],
                        )
                        raise HTTPException(
                            status_code=502,
                            detail=f"Upstream LLM SSE retry error: {stream_error}",
                        ) from stream_error
                    except Exception as stream_error:
                        logger.error("[OpenClaw] upstream SSE retry failed: %s", stream_error, exc_info=True)
                        raise HTTPException(
                            status_code=502,
                            detail=f"Upstream LLM SSE retry failed: {stream_error}",
                        ) from stream_error
                # Retryable upstream error — retry if attempts remain
                if not _is_retryable_status(e.response.status_code) or attempt >= max_retries - 1:
                    logger.error("[OpenClaw] upstream LLM error: %s %s", e.response.status_code, response_text)
                    if not _is_retryable_status(e.response.status_code):
                        # A permanent rejection should surface as itself, not as
                        # a generic gateway error, so the client can act on it.
                        raise HTTPException(status_code=e.response.status_code,
                                            detail=f"Upstream LLM rejected request: {response_text}") from e
                    raise HTTPException(status_code=502, detail=f"Upstream LLM error: {e}") from e
                wait = min(2**attempt + random.uniform(0, 1), 30)
                if e.response.status_code == 429:
                    wait = _retry_after_seconds(e.response, wait)
                logger.warning(
                    "[OpenClaw] upstream LLM error (attempt %d/%d), retrying in %.1fs: %s %s",
                    attempt + 1,
                    max_retries,
                    wait,
                    e.response.status_code,
                    response_text,
                )
                await asyncio.sleep(wait)
                continue
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = min(2**attempt + random.uniform(0, 1), 30)
                    logger.warning(
                        "[OpenClaw] LLM forward failed (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1,
                        max_retries,
                        wait,
                        e,
                    )
                    await asyncio.sleep(wait)
                    continue
                logger.error("[OpenClaw] LLM forward failed: %s", e, exc_info=True)
                raise HTTPException(status_code=502, detail=f"LLM forward error: {e}") from e

    async def _forward_to_llm_bedrock(self, body: dict[str, Any]) -> dict[str, Any]:
        """Forward to AWS Bedrock via BedrockChatClient."""
        from .bedrock_client import BedrockChatClient

        model_id = self.config.llm_model_id
        if not model_id:
            raise HTTPException(
                status_code=503,
                detail="llm.model_id (Bedrock inference profile) is not configured.",
            )

        messages = body.get("messages", [])
        temperature = body.get("temperature", 0.6)
        max_tokens = body.get("max_completion_tokens") or body.get("max_tokens") or 8192

        try:
            client = BedrockChatClient(
                model_id=model_id,
                region=self.config.bedrock_region,
            )
            resp = await asyncio.to_thread(
                client.chat.completions.create,
                model=model_id,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_tokens,
            )
            # Convert BedrockChatClient dataclass response to OpenAI-compatible dict
            choice = resp.choices[0] if resp.choices else None
            return {
                "id": f"chatcmpl-bedrock-{int(time.time())}",
                "object": "chat.completion",
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": choice.message.role if choice else "assistant",
                            "content": choice.message.content if choice else "",
                        },
                        "finish_reason": choice.finish_reason if choice else "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": resp.usage.prompt_tokens,
                    "completion_tokens": resp.usage.completion_tokens,
                    "total_tokens": resp.usage.total_tokens,
                },
            }
        except Exception as e:
            logger.error("[OpenClaw] Bedrock forward failed: %s", e, exc_info=True)
            raise HTTPException(status_code=502, detail=f"Bedrock forward error: {e}") from e

    # ------------------------------------------------------------------ #
    # Session data upload (cloud)                                          #
    # ------------------------------------------------------------------ #

    async def _upload_session_data(
        self,
        session_id: str,
        turns: list[dict],
    ) -> bool:
        """Upload the complete session turn records to cloud storage.

        Session data and skill data live in *separate* cloud paths so they
        can be consumed independently:
          - sessions: ``{group_id}/sessions/{session_id}.jsonl``
          - skills:   ``{group_id}/skills/{name}/SKILL.md``  (handled by SkillHub)
        """
        try:
            from .skill_hub import SkillHub

            hub = SkillHub.object_storage_from_config(self.config)
            if hub is None:
                logger.info(
                    "[SkillHub] session remote upload skipped: no local/OSS/S3 storage configured "
                    "(skill registry may still use Nacos)"
                )
                return False
            session_payload = {
                "session_id": session_id,
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                "user_alias": self.config.sharing_user_alias or os.environ.get("USER", "anonymous"),
                "num_turns": len(turns),
                "turns": turns,
            }

            content = json.dumps(session_payload, ensure_ascii=False)
            oss_key = f"{hub._prefix()}sessions/{session_id}.json"
            await asyncio.to_thread(hub._bucket.put_object, oss_key, content.encode("utf-8"))
            logger.info(
                "[SkillHub] session uploaded: %s (%d turns, %d bytes)",
                oss_key,
                len(turns),
                len(content),
            )
            return True
        except Exception as e:
            logger.warning("[SkillHub] session upload failed: %s", e)
            return False

    def _next_user_turn_num(self, session_id: str) -> int:
        self._user_turn_counts[session_id] = self._user_turn_counts.get(session_id, 0) + 1
        return self._user_turn_counts[session_id]

    def _advance_user_turn_and_maybe_upload(self, session_id: str) -> int:
        user_turn_num = self._next_user_turn_num(session_id)
        self._maybe_upload_session_snapshot(session_id, user_turn_num)
        return user_turn_num

    def _maybe_upload_session_snapshot(self, session_id: str, user_turn_num: int) -> None:
        """Queue a session snapshot when the user-visible turn cadence is reached."""
        interval = max(0, int(getattr(self.config, "sharing_session_upload_interval", 0) or 0))
        if not self.config.sharing_enabled or interval <= 0:
            return
        if user_turn_num <= 0 or user_turn_num % interval != 0:
            return
        turns = copy.deepcopy(self._session_turns.get(session_id, []))
        if not turns:
            return
        self._safe_create_task(self._upload_session_snapshot_and_trigger(session_id, turns))

    async def _upload_session_snapshot_and_trigger(self, session_id: str, turns: list[dict]) -> None:
        uploaded = await self._upload_session_data(session_id, turns)
        if uploaded:
            await self._trigger_evolve()

    async def _trigger_evolve(self) -> None:
        url = str(getattr(self.config, "evolve_server_url", "") or "").strip().rstrip("/")
        if not url:
            return
        import httpx

        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=300.0) as client:
                    resp = await client.post(f"{url}/trigger")
                    resp.raise_for_status()
                    result = resp.json()
                logger.info("[SkillHub] triggered evolve server: %s", url)
                if isinstance(result, dict) and int(result.get("uploaded_skills") or 0) > 0:
                    await self._pull_skills_from_cloud()
                return
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(1.0 * (attempt + 1))
                else:
                    logger.warning("[SkillHub] evolve trigger failed after 3 attempts: %s", e)

    # ------------------------------------------------------------------ #
    # Skill pull (cloud -> local)                                          #
    # ------------------------------------------------------------------ #

    async def _pull_skills_from_cloud(self, skip_names: Optional[set[str]] = None) -> None:
        """Pull latest skills from cloud storage and reload the skill manager.

        This is a *read-only* operation — local skills are never pushed
        automatically.  Use ``skillclaw skills push`` for explicit uploads.

        ``mirror=False`` is mandatory here. This runs unattended (session
        close, the reload poller, the evolve callback), and mirror mode
        rmtree's every local skill absent from the remote manifest. Pointing
        ``local_root`` at a store holding only the skills evolve has written
        would delete the entire local library within seconds. Only an explicit
        ``skillclaw skills pull --mirror`` may delete.
        """
        try:
            from .skill_hub import SkillHub

            hub = SkillHub.from_config(self.config)
            pull_result = await asyncio.to_thread(
                hub.pull_skills,
                self.config.skills_dir,
                mirror=False,
                skip_names=skip_names,
            )
            logger.info(
                "[SkillHub] skill pull: %d downloaded, %d unchanged, %d failed, %d deleted, %d total remote",
                pull_result["downloaded"],
                pull_result["skipped"],
                pull_result.get("failed", 0),
                pull_result.get("deleted", 0),
                pull_result.get("total_remote", 0),
            )
            if pull_result.get("failed_names"):
                logger.warning("[SkillHub] skill pull failed names: %s", ", ".join(pull_result["failed_names"]))
            if self.skill_manager and (
                pull_result.get("downloaded", 0) > 0
                or pull_result.get("deleted", 0) > 0
                or pull_result.get("restored_from_backup", False)
            ):
                self.skill_manager.reload()
        except Exception as e:
            logger.warning("[SkillHub] skill pull failed: %s", e)

    # ------------------------------------------------------------------ #
    # Skill injection                                                      #
    # ------------------------------------------------------------------ #

    def _truncate_messages(
        self,
        messages: list[dict],
        tools,
        max_prompt_tokens: int,
    ) -> list[dict]:
        """Drop oldest non-system messages until the prompt fits the budget.

        The result is always safe to forward upstream: it keeps the original
        message order, keeps a ``user`` turn, and never leaves a ``role:tool``
        result whose issuing assistant turn was dropped.
        """
        return self._truncate_messages_to_budget(messages, tools, max_prompt_tokens, [])

    def _truncate_messages_to_budget(
        self,
        messages: list[dict],
        tools,
        max_prompt_tokens: int,
        leading_system: list[dict],
    ) -> list[dict]:
        def _prompt_len(msgs):
            return _estimate_openai_body_input_tokens({"messages": msgs, "tools": tools})

        original_tokens = _prompt_len(leading_system + messages)
        if original_tokens <= max_prompt_tokens:
            return list(messages)

        # Partition by role without reordering: hoisting mid-conversation
        # system messages to the front used to move a system turn ahead of the
        # user turn it referenced, and left the newest message no longer last.
        sys_msgs: list[dict] = []
        non_sys: list[Any] = []
        for message in messages:
            # A client can POST a bare string as a message. Dropping such
            # entries silently changed behaviour based on prompt size, so they
            # are kept and simply cost tokens.
            if isinstance(message, dict) and message.get("role") == "system":
                sys_msgs.append(message)
            else:
                non_sys.append(message)

        # A role:tool message is only valid when the assistant turn that issued
        # its tool_call_id is still present. Dropping messages one at a time can
        # strip that assistant turn and orphan the result, which OpenAI-compatible
        # upstreams reject with 400 invalid_request_error. Drop whole tool groups.
        units = _message_drop_units(non_sys)

        def fits(candidate: list[Any]) -> bool:
            return _prompt_len(list(leading_system) + list(sys_msgs) + candidate) <= max_prompt_tokens

        # Walk oldest-unit-first and stop as soon as the remainder fits. The
        # newest unit is always kept so the conversation has a live turn.
        dropped = 0
        for step in range(1, len(units)):
            candidate = _drop_oldest_units(non_sys, units, step)
            if not candidate:
                break
            dropped = step
            # A conversation with no `user` turn is rejected upstream, so never
            # settle on a cut that removes every one of them.
            if _has_user_turn(candidate) and fits(candidate):
                break

        result = _drop_oldest_units(non_sys, units, dropped)
        # When even the newest unit is over budget the loop leaves the last
        # user turn dropped; restoring the newest user unit keeps the request
        # forwardable at the cost of a few tokens over the soft limit.
        if not _has_user_turn(result):
            for step in range(len(units), dropped, -1):
                candidate = _drop_oldest_units(non_sys, units, step)
                if candidate and _has_user_turn(candidate):
                    dropped = step
                    result = candidate
                    break
        result = _repair_orphan_tool_results(result)
        result = list(leading_system) + list(sys_msgs) + result
        result_tokens = _prompt_len(result)
        if dropped:
            logger.info(
                "[OpenClaw] context truncated: dropped %d oldest messages (%d -> %d est tokens, limit=%d)",
                dropped,
                original_tokens,
                result_tokens,
                max_prompt_tokens,
            )
        if result_tokens > max_prompt_tokens:
            logger.warning(
                "[OpenClaw] context remains over limit after preserving system messages and the newest message "
                "(%d est tokens, limit=%d)",
                result_tokens,
                max_prompt_tokens,
            )
        return result

    def _inject_skills(self, messages: list[dict]) -> tuple[list[dict], list[str]]:
        """Inject an OpenClaw-compatible skill catalog into the system message.

        Lists ALL eligible skills as an XML ``<available_skills>`` catalog
        with ``<name>``, ``<description>``, and ``<location>`` per entry.
        The model is instructed to ``read`` at most one SKILL.md when
        relevant (lazy loading), matching OpenClaw's injection behaviour.

        Returns (modified_messages, listed_skill_names).
        """
        if not self.skill_manager:
            return messages, []

        try:
            self.skill_manager.refresh_if_changed()
        except Exception as e:
            logger.warning("[SkillManager] failed to refresh local skills: %s", e)

        skill_text = self.skill_manager.build_injection_prompt(
            max_chars=getattr(self.config, "max_skills_prompt_chars", 30_000),
        )
        if not skill_text:
            return messages, []

        all_skills = self.skill_manager.get_all_skills()
        skill_names = [s.get("name", "unknown_skill") for s in all_skills if isinstance(s, dict)]
        logger.info(
            "[SkillManager] listing %d skills in catalog: %s",
            len(skill_names),
            ", ".join(skill_names)[:400],
        )

        self.skill_manager.record_injection(skill_names)

        messages = list(messages)
        sys_indices = [i for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == "system"]
        if sys_indices:
            idx = sys_indices[0]
            existing = _flatten_message_content(messages[idx].get("content", ""))
            messages[idx] = {**messages[idx], "content": existing + "\n\n" + skill_text}
        else:
            messages.insert(0, {"role": "system", "content": skill_text})

        return messages, skill_names

    # ------------------------------------------------------------------ #
    # Turn feedback finalization                                           #
    # ------------------------------------------------------------------ #

    def _maybe_finalize_ready_turns(self, session_id: str):
        """Finalize turns whose optional PRM scoring is done."""
        prm_tasks = self._prm_tasks.setdefault(session_id, {})
        pending = self._pending_turn_data.get(session_id, {})
        for turn_num in sorted(list(pending.keys())):
            prm_task = prm_tasks.get(turn_num)
            if self.config.use_prm and self.prm_scorer:
                if prm_task is None:
                    continue  # waiting for the next turn to provide scoring context
                if not prm_task.done():
                    continue

            turn_data = pending.pop(turn_num)
            prm_result = turn_data.pop("prm_result", None)
            if prm_result is None and prm_task is not None and prm_task.done():
                try:
                    prm_result = prm_task.result()
                except (asyncio.CancelledError, Exception):
                    pass
            # Always drop the task entry once the turn is finalized. Leaving it
            # here (only reachable from the branch above) leaked a completed
            # asyncio.Task per turn for the session's lifetime, and
            # _collect_active_session_ids keys off this dict.
            prm_tasks.pop(turn_num, None)

            self._safe_create_task(
                self._finalize_turn_feedback(
                    turn_num,
                    turn_data,
                    session_id,
                    prm_result,
                )
            )

    async def _finalize_turn_feedback(
        self,
        turn_num: int,
        turn_data: dict[str, Any],
        session_id: str,
        prm_result: Optional[dict],
    ):
        """Finalize a turn after optional PRM scoring.

        SkillClaw acts as an external-agent proxy, so finalization keeps only
        feedback/record side effects that are consumed by the framework.
        """
        score = prm_result.get("score", 0.0) if prm_result else 0.0
        if prm_result:
            self._append_prm_record(session_id, turn_num, score, prm_result.get("votes", []))
            self._session_scored_turns[session_id] = self._session_scored_turns.get(session_id, 0) + 1

        logger.info(
            "[OpenClaw] finalized turn session=%s turn=%d score=%.1f response_chars=%d",
            session_id,
            turn_num,
            score,
            len(turn_data.get("response_text", "")),
        )

    # ------------------------------------------------------------------ #
    # Streaming                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _stream_delta_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Re-key a non-streaming ``tool_calls`` list for streaming consumption.

        The chat completion *message* shape omits ``index``; the streaming
        *delta* shape requires it so a client can attach argument fragments to
        the right call. Forwarding the message shape verbatim makes the openai
        SDK raise ``TypeError: list indices must be integers`` mid-stream.
        """
        indexed: list[dict[str, Any]] = []
        for position, call in enumerate(tool_calls):
            if not isinstance(call, dict):
                continue
            if "index" in call:
                indexed.append(call)
                continue
            indexed.append({"index": position, **call})
        return indexed

    async def _stream_response(self, result: dict[str, Any]):
        payload = result["response"]
        choices = payload.get("choices") or [{}]
        choice = choices[0] if isinstance(choices[0], dict) else {}
        message = choice.get("message") or {}
        if not isinstance(message, dict):
            message = {"content": _flatten_message_content(message)}
        delta: dict[str, Any] = {"role": "assistant", "content": message.get("content", "") or ""}
        if message.get("reasoning_content"):
            delta["reasoning_content"] = message["reasoning_content"]
        if isinstance(message.get("tool_calls"), list) and message["tool_calls"]:
            delta["tool_calls"] = self._stream_delta_tool_calls(message["tool_calls"])
        chunk_base = {
            "id": payload.get("id", ""),
            "object": "chat.completion.chunk",
            "created": payload.get("created") or int(time.time()),
            "model": payload.get("model", ""),
            "session_id": payload.get("session_id", ""),
        }
        first = {**chunk_base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
        final = {
            **chunk_base,
            "choices": [{"index": 0, "delta": {}, "finish_reason": choice.get("finish_reason") or "stop"}],
        }
        yield _sse_chunk(first)
        yield _sse_chunk(final)
        # `stream_options.include_usage` is honoured by emitting the usage-only
        # chunk the spec defines. It carries no choices and must be last.
        usage = payload.get("usage")
        if isinstance(usage, dict) and usage:
            yield _sse_chunk({**chunk_base, "choices": [], "usage": usage})
        yield "data: [DONE]\n\n"

    async def _stream_responses_response(self, response_payload: dict[str, Any]):
        """Yield OpenAI Responses API-compatible SSE events."""
        async for chunk in responses_protocol.stream_response(response_payload):
            yield chunk

    async def _stream_anthropic_response(
        self,
        result: dict[str, Any],
        model: str,
        tool_names: set[str] | None = None,
    ):
        """Yield Anthropic-format SSE events from an internal result dict."""
        async for chunk in anthropic_protocol.stream_from_openai_result(result, model, tool_names):
            yield chunk

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._ready_event.clear()
        self._server_stopped_event.clear()
        cfg = uvicorn.Config(
            self.app,
            host=self.config.proxy_host,
            port=self.config.proxy_port,
            log_level="info",
        )
        self._server = uvicorn.Server(cfg)
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()
        threading.Thread(target=self._print_ready_banner, daemon=True).start()

    def _run_server(self):
        try:
            self._server.run()
        finally:
            self._server_stopped_event.set()
            self._ready_event.clear()

    def _print_ready_banner(self):
        if not self._ready_event.wait(timeout=30):
            return
        if self._server_stopped_event.is_set():
            return
        backend = f"LLM ({self.config.llm_model_id or 'upstream'})"
        banner = (
            f"\n{'=' * 70}\n"
            f"  SkillClaw proxy ready\n"
            f"  proxy {self.config.proxy_host}:{self.config.proxy_port} → {backend}\n"
            f"  Claw agent has been configured to use this proxy automatically.\n"
            f"{'=' * 70}\n"
        )
        logger.info(f"{_GREEN}{banner}{_RESET}")

    def stop(self):
        if self._server is not None:
            self._server.should_exit = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._ready_event.clear()
        self._server_stopped_event.set()

    def wait_until_ready(self, timeout_s: float = 30.0) -> bool:
        return self._ready_event.wait(timeout=timeout_s)

    # ------------------------------------------------------------------ #
    # Utility                                                              #
    # ------------------------------------------------------------------ #

    def _safe_create_task(self, coro):
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)

        def _on_done(t: asyncio.Task):
            self._background_tasks.discard(t)
            self._task_done_cb(t)

        task.add_done_callback(_on_done)
        return task

    @staticmethod
    def _task_done_cb(task: asyncio.Task):
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("[OpenClaw] background task failed: %s", exc, exc_info=exc)
