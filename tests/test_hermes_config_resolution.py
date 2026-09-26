"""Tests for HERMES_CONFIG resolution.

`resolve_hermes_home()` handles the home directory, but Hermes also accepts a
direct path to its config file via HERMES_CONFIG. Sites that read or write
``config.yaml`` (auto-configure, inspect, restore) must honor it, otherwise they
edit a file Hermes never reads.

Order:
    HERMES_CONFIG         (explicit file path, wins)
    <resolved home>/config.yaml
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_STRIP = ("SKILLCLAW_HERMES_HOME", "HERMES_HOME", "HERMES_CONFIG")

_SNIPPET = (
    "from skillclaw._paths import resolve_hermes_config;"
    "print(resolve_hermes_config())"
)


def _resolve(env_overrides: dict) -> str:
    env = {k: v for k, v in os.environ.items() if k not in _STRIP}
    env.update(env_overrides)
    out = subprocess.run(
        [sys.executable, "-c", _SNIPPET],
        capture_output=True, text=True, cwd=REPO, env=env, timeout=60,
    )
    assert out.returncode == 0, f"import failed: {out.stderr[-400:]}"
    return out.stdout.strip()


def test_hermes_config_env_wins():
    got = _resolve({"HERMES_CONFIG": "/tmp/sc-alt/config.yaml"})
    assert got == str(Path("/tmp/sc-alt/config.yaml"))


def test_hermes_config_beats_hermes_home():
    got = _resolve({
        "HERMES_CONFIG": "/tmp/sc-alt/config.yaml",
        "HERMES_HOME": "/tmp/hermes-home",
    })
    assert got == str(Path("/tmp/sc-alt/config.yaml"))


def test_hermes_config_beats_skillclaw_hermes_home():
    got = _resolve({
        "HERMES_CONFIG": "/tmp/sc-alt/config.yaml",
        "SKILLCLAW_HERMES_HOME": "/tmp/sc-home",
    })
    assert got == str(Path("/tmp/sc-alt/config.yaml"))


def test_falls_back_to_home_config_yaml():
    got = _resolve({"HERMES_HOME": "/tmp/hermes-home"})
    assert got == str(Path("/tmp/hermes-home/config.yaml"))


def test_blank_hermes_config_falls_through():
    got = _resolve({"HERMES_CONFIG": "   ", "HERMES_HOME": "/tmp/hermes-home"})
    assert got == str(Path("/tmp/hermes-home/config.yaml"))


def test_stock_default_preserved():
    got = _resolve({})
    assert got == str(Path.home() / ".hermes" / "config.yaml")


def test_claw_adapter_sites_use_the_resolver():
    """No claw_adapter site may rebuild the config path from the home constant."""
    src = (REPO / "skillclaw" / "claw_adapter.py").read_text(encoding="utf-8")
    assert '_HERMES_HOME / "config.yaml"' not in src, (
        "claw_adapter still derives config.yaml from the home constant; "
        "use resolve_hermes_config() so HERMES_CONFIG is honored"
    )
