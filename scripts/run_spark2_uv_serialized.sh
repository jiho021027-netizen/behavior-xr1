#!/usr/bin/env bash
set -euo pipefail

LOCKFILE=${BEHAVIOR_XR1_ISAAC_LOCK:-/tmp/behavior_xr1_isaac.lock}
UV_ROOT=${BEHAVIOR_UV_ROOT:-/home/edgexpert00/projects/behavior-2026-uv}
DATA_ROOT=${OMNIGIBSON_DATA_PATH:-/home/edgexpert00/projects/behavior-2026-validation/datasets}
APPDATA_ROOT=${OMNIGIBSON_APPDATA_PATH:-$UV_ROOT/appdata}

export OMNI_KIT_ACCEPT_EULA=YES
export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1
export OMNIGIBSON_DATA_PATH="$DATA_ROOT"
export OMNIGIBSON_APPDATA_PATH="$APPDATA_ROOT"
export OMNIGIBSON_HEADLESS=1
mkdir -p "$APPDATA_ROOT"

if ! command -v flock >/dev/null 2>&1; then
  echo "ERROR: flock is required for serialized Isaac launches" >&2
  exit 2
fi

exec flock -n "$LOCKFILE" "$@"
