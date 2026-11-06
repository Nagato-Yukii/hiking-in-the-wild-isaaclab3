#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ISAACLAB_PATH="${ISAACLAB_PATH:-$TASK_ROOT/../IsaacLab-3.0.0}"
UV_BIN="${UV_BIN:-uv}"
cd "$ISAACLAB_PATH"
"$UV_BIN" sync --locked --inexact --extra video --extra importers
"$UV_BIN" pip install --python .venv/bin/python tensorboardX==2.6.5
.venv/bin/python "$TASK_ROOT/scripts/fetch_data.py"
.venv/bin/python "$TASK_ROOT/scripts/convert_asset.py"
