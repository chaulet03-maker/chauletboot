#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
while true; do
  python3 -m bot.engine || echo "[run_forever] crashed at $(date)"
  sleep 3
done
