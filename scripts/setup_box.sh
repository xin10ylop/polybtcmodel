#!/bin/bash
# One-shot setup of the paper trader on a fresh Ubuntu box (run as a user with sudo).
# The research scripts hard-code /home/user/... paths, so the repos are placed there.
set -e
SUDO=$(command -v sudo || true); [ "$(id -u)" = 0 ] && SUDO=""        # works as a sudo user or as root
ME=$(id -un)
GH=${GH:-https://github.com/xin10ylop}      # use git@github.com:xin10ylop if you clone over SSH
BR=claude/polymarket-btc-trading-system-ewbqwd
echo "== feed reachability (US IPs are blocked by Binance's main API; the mirrors below must answer)"
curl -sS -m 10 "https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=1s&limit=1" | head -c 80; echo
curl -sS -m 10 "https://www.okx.com/api/v5/market/trades?instId=BTC-USDT-SWAP&limit=1" | head -c 80; echo
echo "== Binance perp (the archive's own feed; HTTP 451 = restricted location, then keep PERP_FEED=okx):"
curl -sS -m 10 -o /dev/null -w "fapi.binance.com -> HTTP %{http_code}\n" "https://fapi.binance.com/fapi/v1/aggTrades?symbol=BTCUSDT&limit=1" || true
curl -sS -m 10 -o /dev/null -w "clob.polymarket.com -> HTTP %{http_code}\n" "https://clob.polymarket.com/markets?limit=1" || true
$SUDO mkdir -p /home/user && $SUDO chown "$ME" /home/user
cd /home/user
for r in btc-predict btc-movement btcmovement2; do [ -d $r ] || git clone $GH/$r.git; done        # model repos: default branch
[ -d polybtcmodel ] || git clone -b $BR $GH/polybtcmodel.git
git -C polybtcmodel pull -q                                         # this copy is the one that runs
$SUDO apt-get install -y -qq git curl python3-venv python3-full libgomp1 > /dev/null   # libgomp1: OpenMP runtime for LightGBM
[ -d /home/user/venv ] || python3 -m venv /home/user/venv
PY=/home/user/venv/bin/python3
$PY -m pip install -q --upgrade pip
$PY -m pip install -q -r polybtcmodel/requirements.txt websocket-client
echo "== 10 days of Binance archive history (spot, ETH, perp) in every format the models need"
START=$(date -u -d "11 days ago" +%F); END=$(date -u -d "yesterday" +%F)
$PY polybtcmodel/scripts/fetch_binance.py $START $END 4
BTC_DATA_DIR=/home/user/data $PY -c "import sys; sys.path.insert(0,'/home/user/btcmovement2'); from btcmodel.timeline import build_timeline; build_timeline()"
echo "== start the paper trader (auto-restart loop, logs in /home/user/paper/, outside git)"
export PAPER_DIR=/home/user/paper; mkdir -p $PAPER_DIR
cd /home/user/polybtcmodel
for pid in $(ps -eo pid,cmd | awk '/[p]aper_trader.py/ {print $1}'); do kill $pid; done; sleep 2    # one trader only
export PERP_FEED=${PERP_FEED:-okx}     # set PERP_FEED=binance on a host where fstream.binance.com is reachable (non-US): the archive-identical feed
nohup bash -c 'while true; do /home/user/venv/bin/python3 -W ignore polybtc/paper_trader.py >> $PAPER_DIR/stdout.log 2>&1; echo "trader exited $(date -u)" >> $PAPER_DIR/stdout.log; sleep 5; done' > /dev/null 2>&1 &
sleep 5; tail -3 $PAPER_DIR/paper.log 2>/dev/null || true
echo "running.  log:   tail -f /home/user/paper/paper.log"
echo "          tally: PAPER_DIR=/home/user/paper /home/user/venv/bin/python3 /home/user/polybtcmodel/polybtc/paper_trader.py --summary"
echo "          audit: PAPER_DIR=/home/user/paper /home/user/venv/bin/python3 /home/user/polybtcmodel/polybtc/paper_audit.py --day $(date -u -d yesterday +%F)"
