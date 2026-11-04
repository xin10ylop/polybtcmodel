#!/bin/bash
# One-shot setup of the paper trader on a fresh Ubuntu box (run as a user with sudo).
# The research scripts hard-code /home/user/... paths, so the repos are placed there.
set -e
GH=${GH:-https://github.com/xin10ylop}      # use git@github.com:xin10ylop if you clone over SSH
BR=claude/polymarket-btc-trading-system-ewbqwd
echo "== feed reachability (US IPs are blocked by Binance's main API; the mirrors below must answer)"
curl -sS -m 10 "https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=1s&limit=1" | head -c 80; echo
curl -sS -m 10 "https://www.okx.com/api/v5/market/trades?instId=BTC-USDT-SWAP&limit=1" | head -c 80; echo
sudo mkdir -p /home/user && sudo chown "$USER" /home/user
cd /home/user
for r in btc-predict btc-movement btcmovement2; do [ -d $r ] || git clone $GH/$r.git; done        # model repos: default branch
[ -d polybtcmodel ] || git clone -b $BR $GH/polybtcmodel.git
python3 -m pip install -q -r polybtcmodel/requirements.txt websocket-client
echo "== 10 days of Binance archive history (spot, ETH, perp) in every format the models need"
START=$(date -u -d "11 days ago" +%F); END=$(date -u -d "yesterday" +%F)
python3 polybtcmodel/scripts/fetch_binance.py $START $END 4
BTC_DATA_DIR=/home/user/data python3 -c "import sys; sys.path.insert(0,'/home/user/btcmovement2'); from btcmodel.timeline import build_timeline; build_timeline()"
echo "== start the paper trader (auto-restart loop, logs in /home/user/polybtcmodel/paper/)"
cd /home/user/polybtcmodel
nohup bash -c 'while true; do python3 -W ignore polybtc/paper_trader.py >> paper/stdout.log 2>&1; echo "trader exited $(date -u)" >> paper/stdout.log; sleep 5; done' > /dev/null 2>&1 &
echo "running. tally:  python3 polybtc/paper_trader.py --summary     log: tail -f paper/paper.log"
