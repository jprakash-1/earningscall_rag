#!/usr/bin/env bash
set -euo pipefail

if [[ ! -x ./.venv/bin/python ]]; then
  echo "Missing .venv Python at ./.venv/bin/python"
  echo "Create environment first: python3 -m venv .venv"
  echo "Then install deps: ./.venv/bin/python -m ensurepip --upgrade && ./.venv/bin/pip install -e ."
  exit 1
f

./.venv/bin/python -m src.data.run "$@"
