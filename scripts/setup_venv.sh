#!/usr/bin/env bash
# Create the project venv with uv (host Python 3.10 has no python3-venv). No sudo needed.
set -euo pipefail
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV="${UV:-$HOME/.local/bin/uv}"
[ -x "$UV" ] || pip install --user uv
"$UV" venv -p /usr/bin/python3.10 --system-site-packages "$WS/.venv"
"$UV" pip install -p "$WS/.venv" -r "$WS/requirements.txt"
