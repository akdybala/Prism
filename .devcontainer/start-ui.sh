#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

if pgrep -f "signal_ui.py --host 0.0.0.0 --port 8000" >/dev/null; then
  echo "Prism Signal Inspector is already running on port 8000."
  exit 0
fi

nohup python signal_ui.py \
  --host 0.0.0.0 \
  --port 8000 \
  --no-browser \
  --skip-warmup \
  > /tmp/prism-signal-ui.log 2>&1 &

echo "Starting Prism Signal Inspector on port 8000."
echo "Logs: /tmp/prism-signal-ui.log"
