#!/usr/bin/env bash
# One competition round in a fresh cloud session: install, decide, upload.
# Needs env vars CODABENCH_TOKEN, TEAM_ID, TEAM_TOKEN (environment secrets).
# Exits quietly with NO_OPEN_ROUND outside a submission window.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m pip install -q -r requirements.txt httpx==0.28.1
[ -d starter-kit ] || python3 scripts/download_data.py --kit-only
mkdir -p private
python3 -m eyris.live run "$@" 2>&1 | grep -v -i "warning" | tee -a "private/cloud_run.log"
