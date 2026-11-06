#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$TASK_ROOT:$TASK_ROOT/vendor/instinct_rl${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1
ISAACLAB_PATH="${ISAACLAB_PATH:-$TASK_ROOT/../IsaacLab-3.0.0}"
UV_BIN="${UV_BIN:-uv}"
cd "$ISAACLAB_PATH"
exec "$UV_BIN" run --locked --inexact --extra video --extra importers --with tensorboardX==2.6.5 python "$TASK_ROOT/scripts/run.py" "$@"
