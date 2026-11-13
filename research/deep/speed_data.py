"""In-window state every 15 seconds for every BTC 5-minute window since Sept 1 (60-second-average settlement): the Up share price now and 10/20/30/60 s ago,
taker flow and volume over the last 10/30 s, Bitcoin against the starting average (price now, 60-s average, the part of the final minute already seen), Bitcoin's
own change over 10/30 s, and what Up could be sold / bought for right after (first sale / first purchase stamped 5-20 s later).  Only prints stamped at or before
the moment are used for the state.     python research/deep/speed_data.py  ->  /home/user/data/deep/speed_BTC.parquet"""
import sys, glob, numpy as np, pandas as pd; sys.path.insert(0, "/home/user/polybtcmodel/research/deep"); sys.path.insert(0, "/home/user/btcmovement2")
from policy_lab import Tix, date, upwin, ok
from btcmodel.timeline import load_timeline
tl = load_timeline(); t0 = int(tl.t0_epoch); sc_ = tl["s_close"]; MK = np.arange(30, 271, 15); sel = ok & np.asarray(date >= "2026-09-01"); W = {int(Tix[i]): bool(upwin[i]) for i in np.flatnonzero(sel)}; out = []
for f in sorted(glob.glob("/home/user/data/polymarket/trades/2026-09-*.parquet") + glob.glob("/home/user/data/polymarket/trades/2026-1*.parquet")):
    t = pd.read_parquet(f, columns=["ts", "side", "outcome_index", "price", "size", "window_start"]); t = t[t.window_start.isin(W)]
    if not len(t): continue
    t["rel"] = t.ts - t.window_start; t = t[(t.rel >= -60) & (t.rel < 300)].sort_values(["window_start", "ts"], kind="stable")
    for w, g in t.groupby("window_start", sort=False):
        w = int(w); a = pd.Series(np.asarray(sc_[w - 960 - t0: w + 300 - t0], dtype=float)).ffill().bfill().values
        if len(a) < 1260: continue
        sig = np.std(np.diff(a[:900])); a = a[900:]; ref = a[:60].mean(); rel = g.rel.values; o0 = g.outcome_index.values == 0; buy = g.side.values == "BUY"; pu = np.where(o0, g.price.values, 1 - g.price.values); lu = (o0 & buy) | (~o0 & ~buy); usd = g.price.values * g["size"].values
        cs = np.concatenate([[0], np.cumsum(usd)]); cf = np.concatenate([[0], np.cumsum(np.where(lu, usd, -usd))]); ix = lambda x: np.searchsorted(rel, x, side="right")        # number of prints stamped <= x
        def last(x): j = ix(x) - 1; return np.where(j >= 0, pu[np.maximum(j, 0)], np.nan)
        hr, hp = rel[~lu], pu[~lu]; br, bp = rel[lu], pu[lu]
        def nxt(r_, p_): q = np.searchsorted(r_, MK + 5, side="left"); okq = q < len(r_); okq[okq] &= r_[q[okq]] < (MK + 5)[okq] + 15; v = np.full(len(MK), np.nan); v[okq] = p_[q[okq]]; return v
        d = {"T": w, "m": MK, "up": W[w], "p": last(MK), "p10": last(MK - 10), "p20": last(MK - 20), "p30": last(MK - 30), "p60": last(MK - 60), "vol10": cs[ix(MK)] - cs[ix(MK - 10)], "vol30": cs[ix(MK)] - cs[ix(MK - 30)], "flow10": cf[ix(MK)] - cf[ix(MK - 10)], "flow30": cf[ix(MK)] - cf[ix(MK - 30)],
             "spot": a[60 + MK - 1] - ref, "avg60": np.array([a[m:60 + m].mean() - ref for m in MK]), "fin": np.array([a[300:60 + m].mean() - ref if m > 240 else 0.0 for m in MK]), "b10": a[60 + MK - 1] - a[60 + MK - 11], "b30": a[60 + MK - 1] - a[60 + MK - 31], "sig": sig, "bid": nxt(hr, hp), "ask": nxt(br, bp)}
        out.append(pd.DataFrame(d))
D = pd.concat(out, ignore_index=True); D.to_parquet("/home/user/data/deep/speed_BTC.parquet", index=False); print(len(D), "rows,", D["T"].nunique(), "windows; sale price known", round(D.bid.notna().mean(), 3), "purchase price known", round(D.ask.notna().mean(), 3))
