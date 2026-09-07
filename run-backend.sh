#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT/backend"
PYTHON="${PYTHON:-python3.11}"
if [[ ! -d .venv ]]; then
  "$PYTHON" -m venv .venv
  .venv/bin/pip install -e .
fi
export PYTHONPATH=.
exec .venv/bin/uvicorn app.main:app --reload --port 8000
