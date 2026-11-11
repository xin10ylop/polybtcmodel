"""Data for the stop-loss study: for every simulated fill of the live setup (T-120, top 30 %, T-45 confirmation, queue-aware, all hours),
at every 30-second mark of the window: how far Bitcoin (Binance spot) is from the window's reference price in our direction, and what our
shares could be sold for at that moment (from the real Polymarket prints).   python research/deep/stoploss_data.py"""
import sys, glob, numpy as np, pandas as pd; sys.path.insert(0, "/home/user/polybtcmodel/research/deep"); sys.path.insert(0, "/home/user/btcmovement2")
from policy_lab import run, Tix, SC, date, upwin, DEEP
from btcmodel.timeline import load_timeline
MARKS = np.arange(30, 300, 30); r = run(dec=-120, tier=0.70, confirm=dict(cancel=True), tp=None, cross="skip", fresh=5, queue=50.0)
I = r["i"]; W = Tix[I]; side = np.where(SC[-120][I] > 0, 0, 1); win = r["v"] > 0; ft = r["ft"]; pos = {int(w): k for k, w in enumerate(W)}
atl = load_timeline(); sc_ = atl["s_close"]; t0 = int(atl.t0_epoch)
def px(t): return float(sc_[t - t0])
TW = pd.Timestamp("2026-09-01").timestamp(); n = len(W); move = np.full((n, len(MARKS)), np.nan); ref = np.full(n, np.nan); fin = np.full(n, np.nan)
for k, w in enumerate(W):
    w = int(w); a = np.asarray(sc_[w - 60 - t0: w + 300 - t0], dtype=float); a = pd.Series(a).ffill().bfill().values
    ref[k] = a[:60].mean() if w >= TW else a[59]                     # reference: 60-s average before the open (settlement rule since Sept 1), the price at the open before
    sgn = 1.0 if side[k] == 0 else -1.0; move[k] = sgn * (a[60 + MARKS - 1] - ref[k]); fin[k] = sgn * ((a[-60:].mean() if w >= TW else a[-1]) - ref[k])
bid = np.full((n, len(MARKS)), np.nan); sell = np.full((n, len(MARKS)), np.nan); liq = np.zeros((n, len(MARKS))); last = np.full((n, len(MARKS)), np.nan)
for f in sorted(glob.glob("/home/user/data/polymarket/trades/*.parquet")):
    t = pd.read_parquet(f, columns=["ts", "side", "outcome_index", "price", "size", "window_start"]); t = t[t.window_start.isin(pos)]
    if not len(t): continue
    t["rel"] = t.ts - t.window_start; t = t[(t.rel >= -20) & (t.rel < 300)].sort_values(["window_start", "ts"], kind="stable")
    for w, g in t.groupby("window_start", sort=False):
        k = pos[int(w)]; rel = g.rel.values; o0 = g.outcome_index.values == 0; buy = g.side.values == "BUY"; ours = o0 if side[k] == 0 else ~o0
        ps = np.where(ours, g.price.values, 1 - g.price.values); hit = (ours & ~buy) | (~ours & buy); usd = g.price.values * g["size"].values      # hit: someone sold our side (what a seller gets)
        for j, m in enumerate(MARKS):
            b = rel < m
            if b.any(): q = np.flatnonzero(b)[-1]; last[k, j] = ps[q]; bid[k, j] = ps[q] if hit[q] else ps[q] - 0.01
            a_ = hit & (rel >= m) & (rel < m + 15)
            if a_.any(): q = np.flatnonzero(a_); sell[k, j] = ps[q[0]]; liq[k, j] = usd[q].sum()
np.savez_compressed(f"{DEEP}/stoploss_BTC.npz", W=W, side=side, win=win, ft=ft, date=np.array(date[I], dtype="datetime64[ns]"), marks=MARKS, move=move, ref=ref, fin=fin, bid=bid, sell=sell, liq=liq, last=last)
print("saved", n, "trades; win", win.mean().round(3), "| final move sign agrees with the outcome:", ((fin > 0) == win).mean().round(3), "| sell price known at T+180:", np.isfinite(sell[:, 5]).mean().round(3), "bid estimate known:", np.isfinite(bid[:, 5]).mean().round(3))
print("median |move| in $ at T+60/120/180/240:", [round(float(np.nanmedian(np.abs(move[:, j]))), 1) for j in (1, 3, 5, 7)], "| median reference price", round(float(np.nanmedian(ref))))
