# Windows PowerShell quickstart — installs uv venv, deps, then runs the smoke test.
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\quickstart.ps1
$ErrorActionPreference = "Stop"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "[quickstart] uv not found — install from https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
}

Write-Host "[quickstart] creating .venv with Python 3.11"
uv venv --python 3.11

Write-Host "[quickstart] activating venv"
. .venv\Scripts\Activate.ps1

Write-Host "[quickstart] installing project (editable)"
uv pip install -e ".[dev]"

if (-not $env:WANDB_API_KEY) {
    Write-Host "[quickstart] WANDB_API_KEY not set — running W&B in offline mode"
    $env:WANDB_MODE = "offline"
}

Write-Host "[quickstart] running smoke test (~30-60 s on single 5090)"
uv run python -m hemefm.train experiment=hello_world

Write-Host ""
Write-Host "[quickstart] done. Check outputs\ for the run directory."
