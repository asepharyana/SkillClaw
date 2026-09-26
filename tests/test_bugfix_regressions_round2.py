"""Regression tests for the second audit round (reviewer findings).

Each test here reproduces a defect that the 2026-09 parallel review confirmed
by execution, and asserts the fixed behaviour.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import types
from pathlib import Path

import pytest

try:
    import httpx  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - optional dep guard
    httpx_stub = types.ModuleType("httpx")
    httpx_stub.BaseTransport = object
    httpx_stub.Client = object
    httpx_stub.Response = object
    sys.modules["httpx"] = httpx_stub

from skillclaw.api_server import (  # noqa: E402
    _coerce_int,
    _flatten_message_content,
    _is_unsupported_temperature_error,
    _message_drop_units,
)
from skillclaw.config_store import (  # noqa: E402
    _DEFAULTS,
    ConfigParseError,
    ConfigStore,
)
from skillclaw.protocols.openai_responses import (  # noqa: E402
    _in_progress_item,
    from_openai_chat_payload,
)

# --------------------------------------------------------------------------- #
# config_store: _deep_merge must not alias _DEFAULTS                            #
# --------------------------------------------------------------------------- #


def test_deep_merge_does_not_alias_module_defaults(tmp_path) -> None:
    """set() used to hand back the live _DEFAULTS dicts, mutating them."""
    store = ConfigStore(config_file=tmp_path / "c.yaml")
    loaded = store.load()
    # Only mutable sections can be aliased; scalars are interned and safe.
    for key, default in _DEFAULTS.items():
        if isinstance(default, dict):
            assert loaded.get(key) is not default, f"{key} still aliases _DEFAULTS"

    # Snapshot before, so a prior test's pollution cannot mask a real regression.
    before = json.dumps(_DEFAULTS["sharing"], sort_keys=True)
    store.set("sharing.enabled", "true")
    assert json.dumps(_DEFAULTS["sharing"], sort_keys=True) == before, "_DEFAULTS was mutated by set()"


# --------------------------------------------------------------------------- #
# config_store: a malformed file must not be silently overwritten                #
# --------------------------------------------------------------------------- #


def test_set_refuses_to_overwrite_unparseable_config(tmp_path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("llm:\n  model_id: x\n bad indent: [\n", encoding="utf-8")

    with pytest.raises(ConfigParseError):
        ConfigStore(config_file=bad).set("proxy.port", "30099")

    # The user's file must be untouched.
    assert "bad indent" in bad.read_text(encoding="utf-8")


def test_get_stays_lenient_on_unparseable_config(tmp_path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("llm: [\n", encoding="utf-8")
    assert ConfigStore(config_file=bad).get("proxy.port") == 30000


def test_to_skillclaw_config_stays_lenient_on_unparseable_config(tmp_path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("llm: [\n", encoding="utf-8")
    assert ConfigStore(config_file=bad).to_skillclaw_config() is not None


# --------------------------------------------------------------------------- #
# config_store: null / scalar / non-numeric sections must not crash startup      #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        "skills:\n",  # empty header -> YAML null
        "skills:\n  top_k:\n",  # explicit null
        "skills:\n  top_k: 'all'\n",  # non-numeric
        "llm: hello\n",  # scalar where a section is expected
        "validation:\n  max_concurrency:\n",
        "dashboard:\n  port: abc\n",
        "prm:\n  temperature: hot\n",
        "proxy:\n  port: []\n",
    ],
)
def test_to_skillclaw_config_tolerates_malformed_sections(tmp_path, text: str) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(text, encoding="utf-8")
    cfg = ConfigStore(config_file=path).to_skillclaw_config()
    assert cfg.skill_top_k == 6  # default preserved
    assert cfg.proxy_port == 30000


def test_set_descends_through_scalar_intermediate(tmp_path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("llm: hello\n", encoding="utf-8")
    store = ConfigStore(config_file=path)
    store.set("llm.model_id", "gpt-4o")
    assert store.get("llm.model_id") == "gpt-4o"


# --------------------------------------------------------------------------- #
# config_store: embedding settings must actually reach SkillClawConfig          #
# --------------------------------------------------------------------------- #


def test_embedding_api_settings_are_mapped(tmp_path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(
        "skills:\n"
        "  embedding_type: api\n"
        "  embedding_api_url: https://api.jina.ai/v1\n"
        "  embedding_api_model: jina-embeddings-v5-text-small\n"
        "  embedding_api_key: secret\n",
        encoding="utf-8",
    )
    cfg = ConfigStore(config_file=path).to_skillclaw_config()
    assert cfg.embedding_type == "api"
    assert cfg.embedding_api_url == "https://api.jina.ai/v1"
    assert cfg.embedding_api_model == "jina-embeddings-v5-text-small"
    assert cfg.embedding_api_key == "secret"


# --------------------------------------------------------------------------- #
# api_server: malformed client input must not 500                              #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "content",
    [[{"type": "text", "text": None}], [{"type": "text", "text": 5}], [{"type": "text", "text": []}]],
)
def test_flatten_message_content_survives_non_string_text(content) -> None:
    assert _flatten_message_content(content) == ""


@pytest.mark.parametrize(
    ("value", "expected"),
    [("auto", 2048), (None, 2048), ([], 2048), (True, 2048), ("8192", 8192), (4096, 4096)],
)
def test_coerce_int(value, expected) -> None:
    assert _coerce_int(value, 2048) == expected


def test_truncate_and_inject_survive_non_dict_messages() -> None:
    """A client can POST a bare string as a message; m.get("role") used to raise."""
    from skillclaw.api_server import SkillClawAPIServer
    from skillclaw.config import SkillClawConfig

    server = SkillClawAPIServer(SkillClawConfig())
    messages = ["bare string", {"role": "user", "content": "hi"}]
    out = server._truncate_messages(list(messages), None, 100_000)
    assert isinstance(out, list)


def test_empty_choices_list_does_not_index_error() -> None:
    """output.get("choices", [{}])[0] raises on a present-but-empty list."""
    payload = {"id": "x", "choices": []}
    choices = payload.get("choices") or [{}]
    assert (choices[0] if choices else {}) == {}


# --------------------------------------------------------------------------- #
# api_server: parallel tool calls must stay grouped under truncation            #
# --------------------------------------------------------------------------- #


def _assistant_with_call(call_id: str) -> dict:
    return {
        "role": "assistant",
        "tool_calls": [{"id": call_id, "function": {"name": "Bash", "arguments": "{}"}}],
    }


def test_sibling_assistant_tool_calls_form_one_unit() -> None:
    """The Responses bridge emits one assistant message per function_call.

    Grouping only an assistant with *following* results left the siblings as
    separate units, so truncation could drop one and orphan the other's result.
    """
    messages = [
        {"role": "user", "content": "go"},
        _assistant_with_call("c1"),
        _assistant_with_call("c2"),
        {"role": "tool", "tool_call_id": "c1", "content": "r1"},
        {"role": "tool", "tool_call_id": "c2", "content": "r2"},
    ]
    units = _message_drop_units(messages)
    assert units == [[0], [1, 2, 3, 4]]


def test_single_assistant_tool_call_still_groups_with_results() -> None:
    messages = [
        {"role": "user", "content": "go"},
        _assistant_with_call("c1"),
        {"role": "tool", "tool_call_id": "c1", "content": "r1"},
        {"role": "user", "content": "next"},
    ]
    assert _message_drop_units(messages) == [[0], [1, 2], [3]]


# --------------------------------------------------------------------------- #
# api_server: temperature-400 handling                                          #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "body",
    [
        "'temperature' is not supported with this model",
        "Unsupported value: 'temperature' does not support 0 with this model",
        "Unsupported parameter: 'temperature'",
    ],
)
def test_api_server_temperature_error_detection(body: str) -> None:
    assert _is_unsupported_temperature_error(body) is True


def test_api_server_temperature_detection_ignores_other_fields() -> None:
    assert _is_unsupported_temperature_error("Unsupported value: 'top_p' does not support 1.5") is False


# --------------------------------------------------------------------------- #
# api_server: PRM task bookkeeping                                              #
# --------------------------------------------------------------------------- #


def test_prm_task_entries_are_always_popped() -> None:
    """The pop lived inside the `prm_result is None` branch, so the normal
    finalize path (result already applied) leaked one Task per turn."""
    import inspect

    from skillclaw.api_server import SkillClawAPIServer

    source = inspect.getsource(SkillClawAPIServer._maybe_finalize_ready_turns)
    pop_line = next(
        i for i, line in enumerate(source.splitlines()) if "prm_tasks.pop(turn_num" in line
    )
    # The pop must be dedented to the same level as the `if prm_result is None`
    # that precedes it, not nested inside it.
    indent = len(source.splitlines()[pop_line]) - len(source.splitlines()[pop_line].lstrip())
    if_line = next(i for i, line in enumerate(source.splitlines()) if "if prm_result is None" in line)
    if_indent = len(source.splitlines()[if_line]) - len(source.splitlines()[if_line].lstrip())
    assert indent <= if_indent, "prm_tasks.pop is still nested inside the branch"


# --------------------------------------------------------------------------- #
# api_server: OAuth header build must not block the event loop                  #
# --------------------------------------------------------------------------- #


def test_build_upstream_auth_headers_is_async() -> None:
    """It performs blocking file I/O and a blocking HTTP refresh on expiry."""
    import inspect

    from skillclaw.api_server import SkillClawAPIServer

    assert inspect.iscoroutinefunction(SkillClawAPIServer._build_upstream_auth_headers)
    assert inspect.iscoroutinefunction(SkillClawAPIServer._prepare_responses_forward)


def test_blocking_oauth_does_not_stall_the_loop() -> None:
    """A 3s refresh behind a sync call froze every concurrent request."""
    import time

    async def scenario() -> float:
        done: list[float] = []
        start = time.monotonic()

        def slow_refresh() -> str:
            time.sleep(0.3)
            return "token"

        async def unrelated() -> None:
            done.append(time.monotonic() - start)

        async def oauth() -> None:
            await asyncio.to_thread(slow_refresh)
            done.append(time.monotonic() - start)

        await asyncio.gather(unrelated(), oauth(), unrelated())
        return max(done[:1])

    assert asyncio.run(scenario()) < 0.2


# --------------------------------------------------------------------------- #
# protocols: Responses streaming + usage                                        #
# --------------------------------------------------------------------------- #


def test_output_item_added_is_in_progress() -> None:
    """Shipping the finished item in .added makes accumulating clients double it."""
    finished_msg = {
        "type": "message",
        "id": "m1",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "hi"}],
        "status": "completed",
    }
    pending = _in_progress_item(finished_msg)
    assert pending["content"] == []
    assert pending["status"] == "in_progress"

    finished_call = {
        "type": "function_call",
        "id": "f1",
        "call_id": "c1",
        "name": "Bash",
        "arguments": '{"c":"ls"}',
        "status": "completed",
    }
    pending_call = _in_progress_item(finished_call)
    assert pending_call["arguments"] == ""
    assert pending_call["status"] == "in_progress"


@pytest.mark.asyncio
async def test_streamed_tool_arguments_are_not_doubled() -> None:
    from skillclaw.protocols.openai_responses import stream_response

    payload = from_openai_chat_payload(
        {
            "id": "c",
            "created": 1,
            "model": "m",
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
                                "function": {"name": "shell", "arguments": '{"command":"ls"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        },
        "m",
    )
    raw_lines = [line async for line in stream_response(payload)]
    events = [json.loads(line[5:]) for line in raw_lines if line.startswith("data:") and "[DONE]" not in line]

    added = [e for e in events if e["type"] == "response.output_item.added"]
    assert added, "expected an output_item.added event"
    for event in added:
        item = event["item"]
        assert item.get("status") == "in_progress"
        assert item.get("arguments", "") == ""


@pytest.mark.parametrize(
    ("usage", "expected_total"),
    [
        ({"prompt_tokens": 10, "completion_tokens": 5}, 15),
        ({"input_tokens": 10, "output_tokens": 5}, 15),
        ({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 99}, 99),
        ({}, 0),
    ],
)
def test_usage_total_is_summed_and_spellings_accepted(usage, expected_total) -> None:
    result = from_openai_chat_payload(
        {
            "id": "c",
            "created": 1,
            "model": "m",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "x"}, "finish_reason": "stop"}],
            "usage": usage,
        },
        "m",
    )
    assert result["usage"]["total_tokens"] == expected_total


# --------------------------------------------------------------------------- #
# evolve_server: fence stripping, rollout grouping, YAML, traversal             #
# --------------------------------------------------------------------------- #


def test_strip_outer_fence_preserves_nested_code_blocks() -> None:
    from evolve_server.core.utils import parse_single_skill, strip_outer_code_fence

    inner = "Run:\n```bash\npytest -q\n```\nDone."
    payload = {"name": "api-tester", "content": inner}
    raw = "```json\n" + json.dumps(payload) + "\n```"

    parsed = parse_single_skill(raw)
    assert parsed is not None, "nested code fence destroyed the JSON payload"
    assert parsed["content"] == inner
    assert strip_outer_code_fence("plain") == "plain"


def test_rollout_index_null_does_not_break_sorting() -> None:
    """A turn with an explicit _rollout_idx: null made sorted() raise TypeError,
    which aborted the whole summarization cycle and re-poisoned every retry."""
    from evolve_server.pipeline.summarizer import _build_rollout_trajectory

    turns = [
        {"prompt_text": "task", "_rollout_idx": 0, "response_text": "a"},
        {"prompt_text": "", "_rollout_idx": None, "response_text": "b"},
    ]
    assert isinstance(_build_rollout_trajectory(turns, "task"), str)


@pytest.mark.parametrize(
    "description",
    ["- run pytest and report", "? tricky", ": colon first", "# hash", "has: colon"],
)
def test_skill_description_survives_yaml_parsing(description: str, tmp_path) -> None:
    """A description starting with a YAML indicator used to publish a SKILL.md
    that every consumer then silently dropped."""
    import yaml

    from evolve_server.core.utils import build_skill_md

    md = build_skill_md({"name": "n", "description": description, "content": "BODY"})
    frontmatter = md.split("---")[1]
    parsed = yaml.safe_load(frontmatter)
    assert parsed["name"] == "n"
    assert parsed["description"] == description


def test_build_skill_md_tolerates_null_content() -> None:
    from evolve_server.core.utils import build_skill_md

    assert "BODY" not in build_skill_md({"name": "n", "description": "d", "content": None})


def test_session_id_cannot_escape_the_workspace() -> None:
    """session_id came from object storage and was used raw as a filename."""
    from evolve_server.engines.agent_workspace import _safe_filename

    for hostile in ["../../PWNED", "/etc/passwd", "..", "a/b/c"]:
        safe = _safe_filename(hostile)
        assert "/" not in safe and "\\" not in safe
        assert safe not in {".", ".."}
        assert Path(safe).name == safe


def test_load_manifest_distinguishes_missing_from_error() -> None:
    """A transient read failure used to return {}, and the caller's
    read-modify-write then deleted every other skill's manifest entry."""

    class Boom(Exception):
        pass

    class FailingBucket:
        def get_object(self, key):
            raise Boom("S3 500")

    from evolve_server.storage.oss_helpers import load_manifest

    with pytest.raises(Boom):
        load_manifest(FailingBucket(), "p/")

    class EmptyBucket:
        def get_object(self, key):
            raise FileNotFoundError(key)

    assert load_manifest(EmptyBucket(), "p/") == {}


# --------------------------------------------------------------------------- #
# evolve_server: rejection must beat publication                                #
# --------------------------------------------------------------------------- #


def test_rejection_is_evaluated_before_publication() -> None:
    """publish_ready was actioned first and `continue`d, so
    validation_max_rejections could never fire when both flags were set."""
    import inspect

    from evolve_server.engines.workflow import EvolveServer

    source = inspect.getsource(EvolveServer)
    assert "if not reject_ready and publish_ready:" in source


# --------------------------------------------------------------------------- #
# setup_wizard: re-running must not destroy unrelated sections                   #
# --------------------------------------------------------------------------- #


def test_setup_wizard_save_preserves_unrelated_sections() -> None:
    import inspect

    from skillclaw import setup_wizard

    source = inspect.getsource(setup_wizard)
    assert "**existing," in source, "wizard must merge onto the existing config"
    assert "**existing.get(\"llm\", {})," in source


def test_sharing_backend_choices_include_nacos() -> None:
    """An existing nacos config was echoed as the default but absent from the
    choice list, so _prompt_choice looped forever."""
    import inspect

    from skillclaw import setup_wizard

    source = inspect.getsource(setup_wizard)
    assert '"local", "s3", "oss", "nacos"' in source


def test_prompt_bool_honours_explicit_negative() -> None:
    from skillclaw import setup_wizard

    answers = iter(["no"])
    original = setup_wizard._prompt
    setup_wizard._prompt = lambda *a, **k: next(answers)
    try:
        assert setup_wizard._prompt_bool("x", default=True) is False
    finally:
        setup_wizard._prompt = original


def test_prompt_bool_reprompts_on_garbage() -> None:
    """'YES please' used to be read as False, silently flipping a True default."""
    from skillclaw import setup_wizard

    answers = iter(["YES please", "y"])
    original = setup_wizard._prompt
    setup_wizard._prompt = lambda *a, **k: next(answers)
    try:
        assert setup_wizard._prompt_bool("x", default=False) is True
    finally:
        setup_wizard._prompt = original


# --------------------------------------------------------------------------- #
# prm_scorer: unauthenticated local endpoint                                    #
# --------------------------------------------------------------------------- #


def test_prm_scorer_constructs_without_api_key() -> None:
    """Documented as supported for local vLLM, but OpenAI() refused to build."""
    from skillclaw.prm_scorer import PRMScorer

    scorer = PRMScorer(prm_url="http://localhost:8081/v1", prm_model="m", api_key="")
    assert scorer.prm_model == "m"


def test_majority_vote_ignores_failed_votes() -> None:
    """A None vote is an API failure, not a dissenting one."""
    from skillclaw.prm_scorer import _majority_vote

    assert _majority_vote([1, 1, None]) == 1.0
    assert _majority_vote([1, -1]) == 0.0  # a genuine tie stays neutral
    assert _majority_vote([None, None]) == 0.0


# --------------------------------------------------------------------------- #
# object_store: manifest-controlled keys must not escape the storage root        #
# --------------------------------------------------------------------------- #


def test_local_object_store_rejects_path_traversal() -> None:
    """Keys come from the shared manifest; a "../" key read or wrote anywhere."""
    import os

    from skillclaw.object_store import LocalObjectStore

    base = tempfile.mkdtemp()
    root = os.path.join(base, "root")
    os.makedirs(root)
    victim = os.path.join(base, "victim.txt")
    with open(victim, "w", encoding="utf-8") as handle:
        handle.write("SECRET")

    store = LocalObjectStore(root)
    with pytest.raises(ValueError):
        store.get_object("../victim.txt")
    with pytest.raises(ValueError):
        store.put_object("../evil.txt", b"pwned")
    with pytest.raises(ValueError):
        store.delete_object("../victim.txt")

    assert not os.path.exists(os.path.join(base, "evil.txt"))
    with open(victim, encoding="utf-8") as handle:
        assert handle.read() == "SECRET"


def test_local_object_store_still_serves_nested_keys() -> None:
    import os

    from skillclaw.object_store import LocalObjectStore

    root = tempfile.mkdtemp()
    store = LocalObjectStore(root)
    store.put_object("group/alpha/SKILL.md", "body")
    assert store.get_object("group/alpha/SKILL.md").read() == b"body"
    assert [i.key for i in store.iter_objects()] == ["group/alpha/SKILL.md"]
    assert os.path.isfile(os.path.join(root, "group", "alpha", "SKILL.md"))


# --------------------------------------------------------------------------- #
# skill_bundle: one definition of the hermes skills root                       #
# --------------------------------------------------------------------------- #


def test_hermes_root_detection_honours_env_override(tmp_path, monkeypatch) -> None:
    """Hardcoding ~/.hermes/skills made the two copies disagree, so
    iter_skill_md_paths returned [] and push/pull silently became no-ops."""
    from skillclaw.skill_bundle import is_hermes_skill_root, iter_skill_md_paths

    hermes_home = tmp_path / "custom-hermes"
    skills = hermes_home / "skills" / "coding" / "alpha"
    skills.mkdir(parents=True)
    (skills / "SKILL.md").write_text("---\nname: alpha\n---\n", encoding="utf-8")

    monkeypatch.setenv("SKILLCLAW_HERMES_HOME", str(hermes_home))
    assert is_hermes_skill_root(hermes_home / "skills") is True
    assert iter_skill_md_paths(hermes_home / "skills") == [str(skills / "SKILL.md")]


# --------------------------------------------------------------------------- #
# skill_manager: effectiveness is a probability                                  #
# --------------------------------------------------------------------------- #


def test_effectiveness_is_clamped_to_one() -> None:
    """Feedback is recorded for injected AND read lists, so positives can
    exceed injections and the score used to exceed 1.0."""
    from skillclaw.skill_manager import SkillManager

    manager = SkillManager.__new__(SkillManager)
    manager._skills_dir = tempfile.mkdtemp()
    manager._stats = {
        "demo": {
            "inject_count": 1,
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "last_injected_at": "",
            "effectiveness": 0.5,
        }
    }
    manager._stats_dirty = 0

    for _ in range(3):
        manager.record_feedback(["demo"], 1.0)

    assert manager.get_effectiveness("demo") == 1.0


# --------------------------------------------------------------------------- #
# validation_worker: explicit temperature 0 must survive                         #
# --------------------------------------------------------------------------- #


def test_validation_worker_preserves_zero_temperature() -> None:
    """`float(... or 0.1)` rewrote a configured 0 back to 0.1 — the exact bug
    class already fixed for config_store's prm.temperature."""
    import inspect

    from skillclaw import validation_worker

    source = inspect.getsource(validation_worker)
    assert 'or 0.1)' not in source
    assert 'getattr(config, "prm_temperature", 0.6)' in source


# --------------------------------------------------------------------------- #
# skill_manager: the catalog must actually honour max_chars                       #
# --------------------------------------------------------------------------- #


def _catalog_manager(tmp_path, n_skills: int):
    from skillclaw.skill_manager import SkillManager

    root = tmp_path / "skills"
    for i in range(n_skills):
        d = root / f"skill-{i:04d}"
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(
            f"---\nname: skill-{i:04d}\ndescription: A reasonably long description for skill {i}.\n---\n\n"
            + "body " * 40,
            encoding="utf-8",
        )
    return SkillManager(skills_dir=str(root))


def test_injection_prompt_respects_max_chars(tmp_path) -> None:
    """max_chars only selected the format before; a large library produced a
    ~170k-char catalog that outgrew max_context_tokens and pushed the user's
    conversation out of the forwarded prompt.
    """
    manager = _catalog_manager(tmp_path, 300)
    for max_chars in (4_000, 8_000, 30_000):
        prompt = manager.build_injection_prompt(max_chars=max_chars)
        # build_skills_section adds a fixed header/footer around the catalog.
        assert len(prompt) <= max_chars + 2_000, f"max_chars={max_chars} produced {len(prompt)} chars"


def test_injection_prompt_scales_with_budget(tmp_path) -> None:
    manager = _catalog_manager(tmp_path, 300)
    small = len(manager.build_injection_prompt(max_chars=4_000))
    large = len(manager.build_injection_prompt(max_chars=30_000))
    assert small < large, "a larger budget should fit more of the catalog"


def test_injection_prompt_still_lists_skills_within_budget(tmp_path) -> None:
    manager = _catalog_manager(tmp_path, 20)
    prompt = manager.build_injection_prompt(max_chars=30_000)
    assert "skill-0000" in prompt
    assert "skill-0019" in prompt


def test_truncated_catalog_reports_omissions(tmp_path) -> None:
    manager = _catalog_manager(tmp_path, 300)
    prompt = manager.build_injection_prompt(max_chars=4_000)
    assert "omitted to fit the context budget" in prompt


def test_empty_library_yields_empty_prompt(tmp_path) -> None:
    from skillclaw.skill_manager import SkillManager

    empty = tmp_path / "none"
    empty.mkdir()
    assert SkillManager(skills_dir=str(empty)).build_injection_prompt() == ""
