#!/usr/bin/env bash
# TASK_820 gate-3 paper loop — run daily shortly after UTC midnight:
# refresh daily bars from Kraken, step the paper book (overlay ON), append report.
# Scheduled by systemd user timer `trader-paper.timer` (00:20 UTC, Persistent).
set -euo pipefail
cd "$(dirname "$0")/.."
source trader_env/bin/activate
python tools/fetch_daily_kraken.py --all
python -m trader paper-step --source db
python -m trader report >> runs/820_paper/daily_report.log
