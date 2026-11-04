#!/bin/bash
# Refit btcmovement2 on 2025-01 .. 2026-07 (walk-forward tag _wf0731) using its own pipeline.
set -e
export BTC_DATA_DIR=/home/user/data BTC_REGIME=1 BTC_TAG=_wf0731
cd /home/user/btcmovement2
echo "== download 2025 $(date)"; python3 -m btcmodel.data 2025-01-01 2026-01-31 6
echo "== timeline $(date)"; python3 -m btcmodel.timeline
echo "== dataset $(date)"; python3 -m btcmodel.dataset
echo "== extra targets $(date)"; python3 scripts/build_targets_extra.py
echo "== regime $(date)"; python3 scripts/build_regime.py
echo "== train walk-forward through 2026-07-31 $(date)"; python3 scripts/train_production.py --end 2026-07-31 --tag _wf0731 --old-thin 2
echo "== DONE $(date)"
