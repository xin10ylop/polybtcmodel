"""Data for the 'buy the dip' test: for every BTC 5-minute window (Jun 1 - Nov 7) and each side, the first moment inside the window (0 .. 270 s after the
open) at which that side's shares could be bought at or below x cents, three ways:
  thru : a resting bid at x is certainly filled (somebody sold that side BELOW x)                      -> maker, pays x, no fee
  touch: somebody sold that side at x or lower (a resting bid at x may or may not have been in front)  -> maker, pays x, no fee (optimistic)
  take : somebody bought that side at x or lower, so it was on offer there                             -> taker, pays that price + fee
  python research/deep/dip_buy_data.py  ->  /home/user/data/deep/dip_buy_BTC.npz"""
import glob, numpy as np, pandas as pd
XS = np.array([0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10]); R = {}
for f in sorted(glob.glob("/home/user/data/polymarket/trades/*.parquet")):
    t = pd.read_parquet(f, columns=["ts", "side", "outcome_index", "price", "size", "window_start"]); t["rel"] = t.ts - t.window_start; t = t[(t.rel >= 0) & (t.rel <= 270)].sort_values(["window_start", "ts"], kind="stable")
    if not len(t): continue
    o0 = t.outcome_index.values == 0; buy = t.side.values == "BUY"; t["pu"] = np.where(o0, t.price.values, 1 - t.price.values); t["lu"] = (o0 & buy) | (~o0 & ~buy); t["usd"] = t.price.values * t["size"].values
    for w, g in t.groupby("window_start", sort=False):
        rel = g.rel.values.astype(np.float32); pu = g.pu.values; lu = g.lu.values; usd = g.usd.values; o = np.full((2, len(XS), 5), np.nan, np.float32)      # t_thru, t_touch, t_take, p_take, usd sold at <= x after the first touch
        for s in (0, 1):
            po = pu if s == 0 else 1 - pu; tb = lu if s == 0 else ~lu
            for k, x in enumerate(XS):
                a = np.where(~tb & (po <= x - 0.005))[0]; b = np.where(~tb & (po <= x + 1e-6))[0]; c = np.where(tb & (po <= x + 1e-6))[0]
                if len(a): o[s, k, 0] = rel[a[0]]
                if len(b): o[s, k, 1] = rel[b[0]]; o[s, k, 4] = usd[b].sum()
                if len(c): o[s, k, 2] = rel[c[0]]; o[s, k, 3] = po[c[0]]
        R[int(w)] = o
W = np.array(sorted(R)); np.savez_compressed("/home/user/data/deep/dip_buy_BTC.npz", W=W, XS=XS, A=np.stack([R[w] for w in W])); print(len(W), "windows")
