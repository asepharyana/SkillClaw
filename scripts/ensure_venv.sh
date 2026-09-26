#!/bin/sh
# Ensure the venv's editable install of SkillClaw still points at this
# working tree and can import its runtime deps.
#
# The venv's `python` is a symlink to the system interpreter, so the install is
# only correct while the editable finder (site-packages/__editable__*.pth) is
# present. Regenerate or replace the venv and that file goes away: imports then
# fall through to the system python, where uvicorn/fastapi do not exist, and
# SkillClaw dies with ModuleNotFoundError on start.
#
# Repair is slow (~17s), so probe first (~30ms) and only reinstall when the
# probe actually fails. Exit non-zero only if the repair also fails, so
# systemd does not start a service that cannot possibly work.
set -eu

REPO="/home/code/SkillClaw"
VENV_PY="$REPO/.venv/bin/python"

if [ ! -x "$VENV_PY" ]; then
    echo "skillclaw-ensure-venv: missing $VENV_PY" >&2
    exit 1
fi

cd "$REPO"

# Fast path: the editable finder exists, and skillclaw + runtime deps import
# from OUTSIDE the repo. Probing from the repo root would import the local
# `skillclaw/` directory by cwd and pass even with the editable install gone —
# which is exactly the state that breaks the systemd-launched process.
if [ -n "$(ls "$REPO"/.venv/lib/python*/site-packages/__editable__.skillclaw-*.pth 2>/dev/null)" ] \
    && (cd / && "$VENV_PY" - <<'PY' >/dev/null 2>&1
import importlib.util as util
import os
import sys

spec = util.find_spec("skillclaw")
if spec is None or not spec.origin:
    sys.exit(1)
# Must resolve into this repo, not a stale copy elsewhere.
if os.path.realpath(os.path.dirname(spec.origin)) != os.path.realpath(
    "/home/code/SkillClaw/skillclaw"
):
    sys.exit(1)
for dep in ("uvicorn", "fastapi", "httpx"):
    if util.find_spec(dep) is None:
        sys.exit(1)
PY
    ); then
    exit 0
fi

echo "skillclaw-ensure-venv: editable install broken, reinstalling..." >&2
"$REPO/.venv/bin/pip" install -e "$REPO" --quiet

# Confirm the repair actually produced a working import before allowing boot.
if "$VENV_PY" -c "import skillclaw.api_server" >/dev/null 2>&1; then
    echo "skillclaw-ensure-venv: repaired" >&2
    exit 0
fi

echo "skillclaw-ensure-venv: repair failed, skillclaw cannot import" >&2
exit 1
