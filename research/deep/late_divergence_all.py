"""The 'looks up, is already down' state in the last minute, on EVERY BTC 5-minute window since Sept 1 (not only our trades), both directions:
the 60-second average of Bitcoin is still on one side of the starting price while the price itself is already on the other side.  Manual idea: buy the side the
PRICE points to (or sell the other).  Entry = what the next buyer of that side really paid (first print stamped 5 s after the moment, within 15 s) + taker fee.
  python research/deep/late_divergence_all.py"""
import sys, glob, numpy as np, pandas as pd; sys.path.insert(0, "/home/user/polybtcmodel/research/deep"); sys.path.insert(0, "/home/user/btcmovement2")
from policy_lab import Tix, date, upwin, ok
from btcmodel.timeline import load_timeline
tl = load_timeline(); t0 = int(tl.t0_epoch); sc_ = tl["s_close"]; MK = np.array([210, 225, 240, 255, 270]); LAG = 5; sel = ok & np.asarray(date >= "2026-09-01"); I = np.flatnonzero(sel); pos = {int(Tix[i]): k for k, i in enumerate(I)}; n = len(I)
spot = np.full((n, len(MK)), np.nan); avg = np.full((n, len(MK)), np.nan)
for k, i in enumerate(I):
    w = int(Tix[i]); a = pd.Series(np.asarray(sc_[w - 60 - t0: w + 300 - t0], dtype=float)).ffill().bfill().values
    if len(a) < 360: continue
    ref = a[:60].mean(); spot[k] = a[60 + MK - 1] - ref; avg[k] = [a[m:60 + m].mean() - ref for m in MK]
ask = np.full((n, len(MK), 2), np.nan)      # what the next buyer of Up / of Down paid
for f in sorted(glob.glob("/home/user/data/polymarket/trades/2026-09-*.parquet") + glob.glob("/home/user/data/polymarket/trades/2026-1*.parquet")):
    t = pd.read_parquet(f, columns=["ts", "side", "outcome_index", "price", "size", "window_start"]); t = t[t.window_start.isin(pos)]
    if not len(t): continue
    t["rel"] = t.ts - t.window_start; t = t[(t.rel >= 200) & (t.rel < 300)].sort_values(["window_start", "ts"], kind="stable")
    for w, g in t.groupby("window_start", sort=False):
        k = pos[int(w)]; rel = g.rel.values; o0 = g.outcome_index.values == 0; buy = g.side.values == "BUY"; pu = np.where(o0, g.price.values, 1 - g.price.values); lu = (o0 & buy) | (~o0 & ~buy)
        for s, msk, px in ((0, lu, pu), (1, ~lu, 1 - pu)):
            hr, hp = rel[msk], px[msk]; q = np.searchsorted(hr, MK + LAG, side="left"); okq = q < len(hr); okq[okq] &= hr[q[okq]] < (MK + LAG)[okq] + 15; ask[k, okq, s] = hp[q[okq]]
up = upwin[I]; day = date.values.astype("datetime64[D]")[I]; rng = np.random.default_rng(7); pd.set_option("display.width", 250); rows = []
for j in (0, 1, 2, 3, 4):
    for a_, b_ in ((0, 0), (5, 5), (10, 5), (10, 10), (20, 10)):
        r = []
        for s in (0, 1):                                  # s = side the PRICE points to: 0 = Up (average still below the start, price above), 1 = Down
            sg = 1 if s == 0 else -1; m = (sg * spot[:, j] >= b_) & (sg * avg[:, j] <= -a_) & np.isfinite(ask[:, j, s]) & ((sg * spot[:, j] > 0) & (sg * avg[:, j] < 0))
            p = ask[m, j, s]; c = p + 0.07 * p * (1 - p); w = up[m] if s == 0 else ~up[m]; r.append(pd.DataFrame({"e": w - c, "w": w, "c": c, "d": day[m]}))
        r = pd.concat(r)
        if len(r) < 30: continue
        g = r.groupby("d").e.sum().values; bs = g[rng.integers(0, len(g), (2000, len(g)))].sum(1) / len(r)
        rows.append({"moment": f"T+{MK[j]}", "average on the other side by": f"${a_}+", "price on this side by": f"${b_}+", "cases": len(r), "per day": round(len(r) / len(np.unique(day)), 1), "the price's side wins %": round(100 * r.w.mean(), 1), "it costs ¢ (fee in)": round(100 * r.c.mean(), 1), "profit ¢/share": round(100 * r.e.mean(), 2), "95% lo": round(100 * np.quantile(bs, .025), 1), "95% hi": round(100 * np.quantile(bs, .975), 1)})
print(f"{n} windows since Sept 1.  Buy the side the PRICE points to while the 60-second average still shows the other side:"); print(pd.DataFrame(rows).to_string(index=False))
