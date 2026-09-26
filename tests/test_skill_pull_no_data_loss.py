"""An unattended skill pull must never delete local skills.

This is a regression guard for a real incident: pointing sharing.local_root
at a store holding only evolve's output, then letting an automatic
mirror-pull run, rmtree'd 1260 of 1216 local skills. `_pull_skills_from_cloud`
and the launcher's auto-pull both call pull_skills() unattended, so both now
pass mirror=False.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillclaw.config import SkillClawConfig
from skillclaw.skill_hub import SkillHub


def _build(tmp_path: Path, n_local: int = 50) -> tuple[Path, Path]:
    skills = tmp_path / "skills"
    store = tmp_path / "store"
    for i in range(n_local):
        d = skills / f"local-{i:03d}"
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(
            f"---\nname: local-{i:03d}\ndescription: local only\n---\nbody\n", encoding="utf-8"
        )
    # The remote store holds exactly one skill — as evolve's output does.
    ev = store / "default" / "skills" / "evolved-one"
    ev.mkdir(parents=True)
    (ev / "SKILL.md").write_text(
        "---\nname: evolved-one\ndescription: from evolve\n---\nnew\n", encoding="utf-8"
    )
    (store / "default" / "manifest.jsonl").write_text(
        json.dumps(
            {
                "name": "evolved-one",
                "skill_id": "x",
                "version": 1,
                "sha256": "y",
                "tree_sha256": "z",
                "format": "bundle_v1",
                "entrypoint": "SKILL.md",
                "files": [],
                "category": "general",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return skills, store


def _hub(skills: Path, store: Path) -> SkillHub:
    cfg = SkillClawConfig(
        skills_dir=str(skills),
        sharing_enabled=True,
        sharing_backend="local",
        sharing_local_root=str(store),
    )
    return SkillHub.from_config(cfg)


def test_unattended_pull_does_not_delete_local_skills(tmp_path) -> None:
    skills, store = _build(tmp_path)
    result = _hub(skills, store).pull_skills(str(skills), mirror=False)

    assert result["deleted"] == 0
    surviving = {p.name for p in skills.iterdir() if p.is_dir()}
    assert len([n for n in surviving if n.startswith("local-")]) == 50


def test_unattended_pull_still_delivers_evolved_skills(tmp_path) -> None:
    skills, store = _build(tmp_path)
    result = _hub(skills, store).pull_skills(str(skills), mirror=False)

    assert result["downloaded"] == 1
    assert (skills / "evolved-one" / "SKILL.md").is_file()


def test_pull_from_cloud_uses_non_mirror_mode() -> None:
    """The proxy's periodic pull is unattended; it must never mirror-delete."""
    import inspect

    from skillclaw.api_server import SkillClawAPIServer

    source = inspect.getsource(SkillClawAPIServer._pull_skills_from_cloud)
    assert "mirror=False" in source


@pytest.mark.anyio
async def test_pull_from_cloud_leaves_local_skills_alone(tmp_path, monkeypatch) -> None:
    """Behavioural guard: drive the real method and count survivors."""
    from skillclaw.api_server import SkillClawAPIServer
    from skillclaw.config import SkillClawConfig

    skills, store = _build(tmp_path, n_local=8)
    cfg = SkillClawConfig(
        skills_dir=str(skills),
        sharing_enabled=True,
        sharing_backend="local",
        sharing_local_root=str(store),
    )

    class Hub:
        def pull_skills(self, skills_dir, *, mirror=True, skip_names=None):
            # Mirror mode against this store would delete all 8 locals.
            if mirror:
                for d in sorted(skills.iterdir()):
                    if d.is_dir() and d.name.startswith("local-"):
                        __import__("shutil").rmtree(d)

    monkeypatch.setattr(SkillHub, "from_config", classmethod(lambda cls, config: Hub()))
    monkeypatch.setattr(SkillClawAPIServer, "_skill_reload_poll_loop", lambda self: _noop(), raising=False)

    server = object.__new__(SkillClawAPIServer)
    server.config = cfg
    server.skill_manager = None
    await server._pull_skills_from_cloud()

    survivors = [p for p in skills.iterdir() if p.is_dir() and p.name.startswith("local-")]
    assert len(survivors) == 8, f"unattended pull deleted {8 - len(survivors)} local skills"


async def _noop():
    return None


def test_launcher_auto_pull_uses_non_mirror_mode() -> None:
    import inspect

    from skillclaw.launcher import SkillClawLauncher

    source = inspect.getsource(SkillClawLauncher)
    assert "pull_skills(cfg.skills_dir, mirror=False)" in source


def test_explicit_mirror_pull_still_deletes(tmp_path) -> None:
    """The capability is not removed — an explicit --mirror still mirrors."""
    skills, store = _build(tmp_path, n_local=5)
    result = _hub(skills, store).pull_skills(str(skills), mirror=True)

    assert result["deleted"] == 5
    remaining = {p.name for p in skills.iterdir() if p.is_dir()}
    assert remaining == {"evolved-one"}


def test_mirror_pull_refuses_when_backup_is_incomplete(tmp_path, monkeypatch) -> None:
    """A silently partial backup must block deletion, not enable it."""
    skills, store = _build(tmp_path, n_local=6)
    real_copytree = __import__("shutil").copytree

    def flaky_copytree(src, dst, *args, **kwargs):
        real_copytree(src, dst, *args, **kwargs)
        # Simulate a backup that silently omitted some skills.
        for child in sorted(Path(dst).iterdir())[:3]:
            if child.is_dir():
                __import__("shutil").rmtree(child, ignore_errors=True)

    monkeypatch.setattr("skillclaw.skill_hub.shutil.copytree", flaky_copytree)
    result = _hub(skills, store).pull_skills(str(skills), mirror=True)

    assert result["deleted"] == 0
    assert len([p for p in skills.iterdir() if p.is_dir()]) == 6
