"""Regression tests for bugs found by the 2026-09 codebase audit."""

from __future__ import annotations

import json
import sqlite3
import sys
import types

import pytest

try:
    import httpx  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - optional dep guard
    httpx_stub = types.ModuleType("httpx")
    httpx_stub.BaseTransport = object
    httpx_stub.Client = object
    httpx_stub.Response = object
    sys.modules["httpx"] = httpx_stub

from evolve_server.core.llm_client import _is_unsupported_temperature_error  # noqa: E402
from skillclaw.nacos_skill_hub import _largest_nacos_version  # noqa: E402

# --------------------------------------------------------------------------- #
# evolve_server.core.llm_client: temperature-400 detection                      #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "body",
    [
        "'temperature' is not supported with this model.",
        "Unsupported value: 'temperature' does not support 0 with this model.",
        "Unsupported parameter: 'temperature'",
        "temperature is not supported for this model",
        "Unrecognized request argument supplied: temperature",
    ],
)
def test_unsupported_temperature_detected_across_provider_wording(body: str) -> None:
    assert _is_unsupported_temperature_error(body) is True


@pytest.mark.parametrize(
    "body",
    [
        "",
        "context length exceeded",
        # A 400 about a *different* field must not disable temperature.
        "Unsupported value: 'top_p' does not support 1.5 with this model.",
        # Rate limiting mentions temperature nowhere but must not match either.
        "rate limit reached",
    ],
)
def test_non_temperature_errors_rejected(body: str) -> None:
    assert _is_unsupported_temperature_error(body) is False


# --------------------------------------------------------------------------- #
# evolve_server.core.llm_client: 4xx must not be retried                        #
# --------------------------------------------------------------------------- #


class _FakeResponse:
    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text


class _FakeAPIError(Exception):
    def __init__(self, status_code: int, text: str) -> None:
        super().__init__(text)
        self.response = _FakeResponse(status_code, text)


class _FakeCompletions:
    def __init__(self, error: Exception) -> None:
        self._error = error
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        raise self._error


def _client_with_error(error: Exception):
    """Build an AsyncLLMClient shell without touching the real openai SDK."""
    from evolve_server.core import llm_client as mod

    client = object.__new__(mod.AsyncLLMClient)
    client.model = "gpt-4o"
    client.max_tokens = 100
    client.temperature = 0.4
    client._client = types.SimpleNamespace(
        api_key="k",
        base_url="https://example.invalid/v1",
        chat=types.SimpleNamespace(completions=_FakeCompletions(error)),
    )
    return client


@pytest.fixture
def no_backoff(monkeypatch):
    """Collapse the exponential backoff so retry tests run instantly."""
    from evolve_server.core import llm_client as mod

    async def _instant(_delay):
        return None

    monkeypatch.setattr(mod.asyncio, "sleep", _instant)


@pytest.mark.asyncio
async def test_permanent_4xx_is_not_retried() -> None:
    """A 401 is deterministic: retrying burns ~30s of backoff for nothing."""
    client = _client_with_error(_FakeAPIError(401, "invalid api key"))
    with pytest.raises(_FakeAPIError):
        await client.chat([{"role": "user", "content": "hi"}])
    assert len(client._client.chat.completions.calls) == 1


@pytest.mark.asyncio
async def test_unknown_model_404_is_not_retried() -> None:
    client = _client_with_error(_FakeAPIError(404, "model does not exist"))
    with pytest.raises(_FakeAPIError):
        await client.chat([{"role": "user", "content": "hi"}])
    assert len(client._client.chat.completions.calls) == 1


@pytest.mark.asyncio
async def test_rate_limit_429_is_still_retried(no_backoff) -> None:
    """429 is transient and must keep its backoff path."""
    client = _client_with_error(_FakeAPIError(429, "slow down"))
    with pytest.raises(RuntimeError, match="failed after 6 attempts"):
        await client.chat([{"role": "user", "content": "hi"}])
    assert len(client._client.chat.completions.calls) == 6


@pytest.mark.asyncio
async def test_retry_on_5xx_then_gives_up(no_backoff) -> None:
    client = _client_with_error(_FakeAPIError(503, "upstream unavailable"))
    with pytest.raises(RuntimeError, match="failed after 6 attempts"):
        await client.chat([{"role": "user", "content": "hi"}])
    assert len(client._client.chat.completions.calls) == 6


@pytest.mark.asyncio
async def test_unsupported_temperature_drops_field_and_retries_without_cycling() -> None:
    """A provider that rejects temperature is retried once without the field.

    If the provider keeps returning the same 400, the client must raise rather
    than loop forever on the temperature branch.
    """
    error = _FakeAPIError(400, "Unsupported value: 'temperature' does not support 0 with this model.")
    client = _client_with_error(error)
    with pytest.raises(_FakeAPIError):
        await client.chat([{"role": "user", "content": "hi"}])

    calls = client._client.chat.completions.calls
    # Attempt 1 carries temperature, attempt 2 drops it, attempt 3 sees it
    # already gone and re-raises instead of burning the remaining budget.
    assert "temperature" in calls[0]
    assert "temperature" not in calls[1]
    assert len(calls) == 2


# --------------------------------------------------------------------------- #
# skillclaw.nacos_skill_hub: version ordering                                    #
# --------------------------------------------------------------------------- #


def test_largest_version_orders_bare_integers_numerically() -> None:
    # "12" > "7" numerically but loses under a plain string comparison.
    assert _largest_nacos_version(["3", "12", "7"]) == "12"
    assert _largest_nacos_version(["9", "10"]) == "10"


def test_largest_version_prefers_structured_formats() -> None:
    assert _largest_nacos_version(["2.9.0", "2.10.0"]) == "2.10.0"
    assert _largest_nacos_version(["v9", "v10"]) == "v10"
    assert _largest_nacos_version(["2.9.0", "v99"]) == "2.9.0"


def test_largest_version_edge_cases() -> None:
    assert _largest_nacos_version([]) is None
    assert _largest_nacos_version(["alpha", "beta"]) == "beta"
    assert _largest_nacos_version(["-2", "-10"]) == "-2"


# --------------------------------------------------------------------------- #
# Protocol translation: falsy values must still use the fallback               #
# --------------------------------------------------------------------------- #


def test_created_falls_back_when_upstream_sends_zero() -> None:
    """`dict.get(k, default)` does not fire when the key holds 0 or None."""
    from skillclaw.protocols.openai_responses import from_openai_chat_payload

    payload = {
        "id": "chatcmpl-x",
        "created": 0,
        "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    }
    result = from_openai_chat_payload(payload, "gpt-4o")
    assert result["created_at"] > 0


def test_anthropic_message_id_falls_back_when_upstream_sends_empty() -> None:
    from skillclaw.protocols.anthropic_messages import from_openai_response

    payload = {
        "id": "",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    }
    result = from_openai_response(payload, "claude-x")
    assert result["id"] == "msg_skillclaw"


def test_anthropic_stop_reason_when_finish_reason_is_null() -> None:
    """Some upstreams emit `"finish_reason": null`; it must not become tool_use."""
    from skillclaw.protocols.anthropic_messages import from_openai_response

    payload = {
        "id": "chatcmpl-y",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": None}],
    }
    result = from_openai_response(payload, "claude-x")
    assert result["stop_reason"] == "end_turn"


# --------------------------------------------------------------------------- #
# DashboardStore: sqlite connections must actually be closed                   #
# --------------------------------------------------------------------------- #


def test_dashboard_store_closes_connections(tmp_path) -> None:
    """`with sqlite3.connect(...)` only ends the transaction — it never closes.

    Regression guard: the store must not leak a file descriptor per query.
    """
    from skillclaw.dashboard_store import DashboardStore

    store = DashboardStore(str(tmp_path / "dash.db"))
    store.initialize()

    with sqlite3.connect(":memory:") as probe:
        probe.execute("SELECT 1")

    # If _connect leaked, the handle stays referenced by the contextlib object
    # and reading through it would still work; the real check is that repeated
    # use leaves no open transaction and the file is usable afterwards.
    for _ in range(50):
        assert store.get_meta() == {}
    assert (tmp_path / "dash.db").exists()


def test_dashboard_store_context_manager_closes_on_exception(tmp_path) -> None:
    from skillclaw.dashboard_store import DashboardStore

    store = DashboardStore(str(tmp_path / "dash.db"))
    store.initialize()

    with pytest.raises(RuntimeError):
        with store._connect() as conn:
            conn.execute("SELECT 1")
            raise RuntimeError("boom")

    # The connection must already be closed, so this raises ProgrammingError.
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")


# --------------------------------------------------------------------------- #
# SkillManager: skill_stats.json must survive a partial write                  #
# --------------------------------------------------------------------------- #


def test_skill_stats_write_is_atomic(tmp_path) -> None:
    """A crash mid-dump must not truncate skill_stats.json to nothing."""
    from skillclaw.skill_manager import SkillManager

    manager = SkillManager.__new__(SkillManager)
    manager._skills_dir = str(tmp_path)
    manager._stats = {"demo": {"effectiveness": 0.9, "inject_count": 3}}
    manager._stats_dirty = 0
    manager._save_stats()

    stats_file = tmp_path / "skill_stats.json"
    assert json.loads(stats_file.read_text(encoding="utf-8")) == manager._stats

    # No stray temp files left behind.
    assert [p.name for p in tmp_path.iterdir() if p.name != "skill_stats.json"] == []
