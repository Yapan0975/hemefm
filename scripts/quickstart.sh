#!/usr/bin/env bash
# POSIX quickstart — installs uv venv, deps, then runs the smoke test.
# Usage:  bash scripts/quickstart.sh
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
    echo "[quickstart] uv not found — install from https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

echo "[quickstart] creating .venv with Python 3.11"
uv venv --python 3.11

# shellcheck source=/dev/null
source .venv/bin/activate

echo "[quickstart] installing project (editable)"
uv pip install -e ".[dev]"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "[quickstart] WANDB_API_KEY not set — running W&B in offline mode"
    export WANDB_MODE=offline
fi

echo "[quickstart] running smoke test (~30-60 s on single 5090)"
uv run python -m hemefm.train experiment=hello_world

echo ""
echo "[quickstart] done. Check outputs/ for the run directory."
