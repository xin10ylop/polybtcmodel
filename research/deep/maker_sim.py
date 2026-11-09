"""Realistic maker entry for the pre-open rule, from the Polymarket prints with their aggressor side.
A resting BUY of our token at `bid`, placed at T-45, is filled by the first print in which the aggressor trades AGAINST it at a price
that reaches it:  taker SELL of our token at <= bid,  or  taker BUY of the other token at >= 1 - bid  (the engine mints the pair).
  exact   : first such print (we are assumed to be first in the queue at our price)
  through : first such print strictly better than our bid (the level traded through us: filled whatever the queue)
Stored per window x side x bid: fill time (s relative to T) for both definitions, whether the sell limit `tp` printed after the fill,
and the $ notional of qualifying prints up to each cancel time (capacity).  Exits as everywhere: sell at tp (any print of our side
>= tp after the fill), else settlement.
  ASSET=BTC python research/deep/maker_sim.py   ->  /home/user/data/deep/maker_<asset>.npz"""
import os, sys, glob, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/polybtcmodel/research/deep"); from common import SRC, DEEP  # noqa: E402
ASSET = os.environ.get("ASSET", "BTC"); TP = {"BTC": 0.85, "ETH": 0.80}[ASSET]
BIDS = np.array([0.47, 0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.54, 0.55]); CANCELS = np.array([-40, -30, -15, 0]); T0 = -45
tdir = "/home/user/data/polymarket/trades" if ASSET == "BTC" else "/home/user/data_eth/polymarket/trades"
mk = pd.concat([pd.read_parquet(m) for m, _, _ in SRC[ASSET]["markets"]]).drop_duplicates("window_start").set_index("window_start"); mk = mk[mk.outcome_up_price.isin([0.0, 1.0])]
Ws, FILL, TPH, USD, WIN = [], [], [], [], []
for f in sorted(glob.glob(f"{tdir}/*.parquet")):
    t = pd.read_parquet(f, columns=["ts", "side", "outcome_index", "price", "size", "window_start"]); t = t[t.window_start.isin(mk.index)].sort_values(["window_start", "ts"], kind="stable")
    if len(t) == 0: continue
    for w, g in t.groupby("window_start", sort=False):
        rel = (g.ts.values - w).astype(np.int32); out = g.outcome_index.values; px = g.price.values; buy = g.side.values == "BUY"; usd = px * g["size"].values
        fill = np.full((2, len(BIDS), 2), np.nan, np.float32); tph = np.zeros((2, len(BIDS), 2), bool); cap = np.zeros((2, len(BIDS), len(CANCELS)), np.float32)
        for s in (0, 1):
            ps = np.where(out == s, px, 1 - px)                                    # price of OUR token implied by the print
            against = ((out == s) & ~buy) | ((out != s) & buy)                     # aggressor sells our token / buys the other one
            live = (rel >= T0) & (rel <= 0) & against; sufmax = np.maximum.accumulate(ps[::-1])[::-1]      # max of our token's price from each print on
            for b, bid in enumerate(BIDS):
                for k, cond in enumerate((live & (ps <= bid + 1e-9), live & (ps < bid - 1e-9))):
                    j = np.flatnonzero(cond)
                    if len(j):
                        fill[s, b, k] = rel[j[0]]; nxt = j[0] + 1; tph[s, b, k] = bool(nxt < len(ps) and (sufmax[nxt:][rel[nxt:] <= 300].max(initial=0.0) >= TP))
                q = live & (ps <= bid + 1e-9)
                for c, cn in enumerate(CANCELS): cap[s, b, c] = usd[q & (rel <= cn)].sum()
        Ws.append(w); FILL.append(fill); TPH.append(tph); USD.append(cap); WIN.append(mk.at[w, "outcome_up_price"] == 1.0)
    print(os.path.basename(f), len(Ws), flush=True)
np.savez_compressed(f"{DEEP}/maker_{ASSET}.npz", W=np.array(Ws, np.int64), fill=np.array(FILL), tph=np.array(TPH), usd=np.array(USD), up=np.array(WIN), bids=BIDS, cancels=CANCELS, tp=TP)
print("saved", len(Ws))
