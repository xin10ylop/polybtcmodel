"""'Cheap shares while Bitcoin is on that side': for every BTC window and each side, the first print (5 .. 270 s after the open) at which the side could be
bought at <= x cents WHILE Bitcoin (Binance spot) was more than zmin dollars on that side of the reference price, with the Bitcoin reading taken L seconds
BEFORE the print's timestamp.  The timestamp of a print is the block time, i.e. some seconds after the trade was agreed, so L = 0 looks into the future;
the lags show how fast the effect disappears.   take = somebody bought there (taker, pays price + fee), thru = somebody sold below x (resting bid filled).
  python research/deep/dip_gap_data.py  ->  /home/user/data/deep/dip_gap_BTC.npz"""
import sys, glob, numpy as np, pandas as pd; sys.path.insert(0, "/home/user/btcmovement2")
from btcmodel.timeline import load_timeline
XS = np.array([0.45, 0.40, 0.35, 0.30, 0.25, 0.20]); ZM = np.array([0.0, 10.0, 20.0, 30.0]); LG = np.array([0, 1, 2, 3, 5, 8, 12]); TW = pd.Timestamp("2026-09-01").timestamp()
tl = load_timeline(); sc_ = tl["s_close"]; t0 = int(tl.t0_epoch); R = {}
for f in sorted(glob.glob("/home/user/data/polymarket/trades/*.parquet")):
    t = pd.read_parquet(f, columns=["ts", "side", "outcome_index", "price", "size", "window_start"]); t["rel"] = t.ts - t.window_start; t = t[(t.rel >= 5) & (t.rel <= 270)].sort_values(["window_start", "ts"], kind="stable")
    if not len(t): continue
    o0 = t.outcome_index.values == 0; buy = t.side.values == "BUY"; t["pu"] = np.where(o0, t.price.values, 1 - t.price.values); t["lu"] = (o0 & buy) | (~o0 & ~buy)
    for w, g in t.groupby("window_start", sort=False):
        w = int(w)
        if w - 80 - t0 < 0 or w + 301 - t0 > tl.n: continue
        a = pd.Series(np.asarray(sc_[w - 60 - t0: w + 301 - t0], dtype=float)).ffill().bfill().values; up = a - (a[:60].mean() if w >= TW else a[59])        # dollars above the reference, index = seconds since T-60
        rel = g.rel.values.astype(int); pu = g.pu.values; lu = g.lu.values; o = np.full((2, 2, len(XS), len(ZM), len(LG), 2), np.nan, np.float32)             # side, how (take / thru), x, zmin, lag, (time, price)
        for s in (0, 1):
            po = pu if s == 0 else 1 - pu; tb = lu if s == 0 else ~lu; sg = 1 if s == 0 else -1
            for li, L in enumerate(LG):
                z = sg * up[60 + rel - L]
                for zi, zm in enumerate(ZM):
                    zok = z > zm
                    if not zok.any(): continue
                    for k, x in enumerate(XS):
                        c = np.flatnonzero(zok & tb & (po <= x + 1e-6))
                        if len(c): o[s, 0, k, zi, li] = (rel[c[0]], po[c[0]])
                        c = np.flatnonzero(zok & ~tb & (po <= x - 0.005))
                        if len(c): o[s, 1, k, zi, li] = (rel[c[0]], x)
        R[w] = o
W = np.array(sorted(R)); np.savez_compressed("/home/user/data/deep/dip_gap_BTC.npz", W=W, XS=XS, ZM=ZM, LG=LG, A=np.stack([R[w] for w in W])); print(len(W), "windows")
