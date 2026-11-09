"""Maker fills by time bucket (generalises the first maker simulator): for every window x side x bid, the first print that would have
filled a resting BUY of our token at `bid` inside each bucket of the pre-open timeline, with the aggressor side taken into account
(taker SELL of our token at <= bid, or taker BUY of the other token at >= 1 - bid), plus whether the sell limit printed afterwards
and the $ notional of the qualifying prints in the bucket.  'through' = strictly better than our bid.
Buckets (seconds relative to the nominal open T): [-600,-300) [-300,-180) [-180,-120) [-120,-90) [-90,-60) [-60,-45) [-45,-30) [-30,-15) [-15,0]
  ASSET=BTC python research/deep/maker_sim2.py  ->  /home/user/data/deep/maker2_<asset>.npz"""
import os, sys, glob, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/polybtcmodel/research/deep"); from common import SRC, DEEP  # noqa: E402
ASSET = os.environ.get("ASSET", "BTC"); TP = {"BTC": 0.85, "ETH": 0.80}[ASSET]
BIDS = np.array([0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.54]); EDGES = np.array([-600, -300, -180, -120, -90, -60, -45, -30, -15, 1]); NB = len(EDGES) - 1
tdir = "/home/user/data/polymarket/trades" if ASSET == "BTC" else "/home/user/data_eth/polymarket/trades"
mk = pd.concat([pd.read_parquet(m) for m, _, _ in SRC[ASSET]["markets"]]).drop_duplicates("window_start").set_index("window_start"); mk = mk[mk.outcome_up_price.isin([0.0, 1.0])]
Ws, FILL, TPH, USD, WIN = [], [], [], [], []
for f in sorted(glob.glob(f"{tdir}/*.parquet")):
    t = pd.read_parquet(f, columns=["ts", "side", "outcome_index", "price", "size", "window_start"]); t = t[t.window_start.isin(mk.index)].sort_values(["window_start", "ts"], kind="stable")
    if len(t) == 0: continue
    for w, g in t.groupby("window_start", sort=False):
        rel = (g.ts.values - w).astype(np.int32); out = g.outcome_index.values; px = g.price.values; buy = g.side.values == "BUY"; usd = px * g["size"].values
        bk = np.searchsorted(EDGES, rel, side="right") - 1; inb = (bk >= 0) & (bk < NB); post = rel <= 300
        fill = np.full((2, len(BIDS), NB, 2), np.nan, np.float32); tph = np.zeros((2, len(BIDS), NB, 2), bool); cap = np.zeros((2, len(BIDS), NB), np.float32)
        for s in (0, 1):
            ps = np.where(out == s, px, 1 - px); against = (((out == s) & ~buy) | ((out != s) & buy)) & inb
            sufmax = np.maximum.accumulate(np.where(post, ps, 0.0)[::-1])[::-1]
            for b, bid in enumerate(BIDS):
                q = against & (ps <= bid + 1e-9)
                if not q.any(): continue
                np.add.at(cap[s, b], bk[q], usd[q])
                for k, cond in enumerate((q, against & (ps < bid - 1e-9))):
                    j = np.flatnonzero(cond)
                    if not len(j): continue
                    for jj in j[np.unique(bk[j], return_index=True)[1]]:                 # first qualifying print in each bucket
                        fill[s, b, bk[jj], k] = rel[jj]; tph[s, b, bk[jj], k] = bool(jj + 1 < len(ps) and sufmax[jj + 1] >= TP)
        Ws.append(w); FILL.append(fill); TPH.append(tph); USD.append(cap); WIN.append(mk.at[w, "outcome_up_price"] == 1.0)
    print(os.path.basename(f), len(Ws), flush=True)
np.savez_compressed(f"{DEEP}/maker2_{ASSET}.npz", W=np.array(Ws, np.int64), fill=np.array(FILL), tph=np.array(TPH), usd=np.array(USD), up=np.array(WIN), bids=BIDS, edges=EDGES, tp=TP)
print("saved", len(Ws))
